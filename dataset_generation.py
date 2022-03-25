#
# dataset_generation.py
#
# Generates TalesSkits dataset given data in the form of SFX-free*
# video recordings of skits from various Tales of video games. 
#
# *If SFX are present, there are special requirements.
#
# The skits recorded must have the following qualities:
#   1. The speaker name must be displayed
#   2. The sound and the subtitles must sync up relatively (at no 
#      point should sound of a different speaker be playing audible
#      while another speaker's utterance is on the screen)
#   3. The audio should be clean. No stuttering or artifacts should be
#      present.
#   4. *If SFX is enabled, no text should be visible within the
#      the lower portion of the screen where subtitles normally appear
#      during non-voice SFX instances. VAD theoretically should 
#      deactivate these frames, but this is not a guarantee.  
#
# Ensure that audio configuration hyperparameters match those of the
# project that you want to apply this dataset to. 

from data_params import *
from audio_utils import *
from frame_utils import *
from frame_processing import *
from text_cleaner import *
from speaker_whitelist import speaker_whitelist

# If enabled, will try to include multispeaker_synthesis code for
# speaker verification as a post-processing step. 
if speaker_verification_enabled is True:
  from speaker_verification import *

from multiprocessing import Pool
from functools import partial
from typing import Optional
import cv2
import os
from tqdm import tqdm
import argparse
from unidecode import unidecode

def extract_tales_skits(visualization: Optional[bool] = False, 
                        multiprocessing: Optional[bool] = True):
  """
  Principal function extracting utterances by speaker with transcripts
  from Tales of video game skit recordings. 
  """
  if not os.path.exists(data_folder):
    print("[ERROR] Dataset - Unable to find %s." % data_folder)
  
  # Process every single video, giving each video it's own identifier.
  data_contents = os.listdir(data_folder)
  video_files = []
  for file in data_contents:
    if file.endswith(video_suffix):
      video_files.append(file)

  data_count = len(video_files)
  print("[INFO] Dataset - found %d %s files in folder %s." % (data_count, video_suffix, data_folder))

  video_infos = []
  for video_id in range(initial_video_id_index, data_count+initial_video_id_index):   
    video_filename = video_files[video_id-initial_video_id_index]
    video_fpath = data_folder + "/" + video_filename
    game_name = _determine_game_title(video_filename)

    if multiprocessing:
      video_infos.append((video_id, video_fpath, game_name))
    else:
      _process_skit_video((video_id, video_fpath, game_name), visualization, False)
  
  # Multiprocessing. 
  if len(video_infos) > 0:
    func = partial(_process_skit_video, visualization=visualization, multiprocessing=True)
    job = Pool(n_processes).imap(func, video_infos)
    list(tqdm(job, "Videos Processed", len(video_infos), unit = "video"))

def _determine_game_title(filename):
  """
  Given the filename, retrieve the game title code. This will help
  allow us to behave differently depending on the game skit format. 
  """
  lower_filename = filename.lower()
  for game_name in subtitle_roi_by_game:
    if game_name in lower_filename:
      return game_name

def _process_skit_video(video_info, visualization, multiprocessing=False):
  """
  Processes an entire skit video. For each selected frame in the video,
  takes the following steps:

  1. If the frame does not have any associated audio activity, skip.
  2. Execute OCR on a specific region of the video (subtitles). If no
     text is returned, skip. 
  3. Compare the obtained text to the previous frame. If the text is 
     the same, skip. 
  4. For each new utterance detected, save the old utterance to file 
     (if the transcript is valid). Start a new utterance by processing
     transcript. Skip the transcript if not valid. 
       a. For each transcript, find the speaker name. This must be 
          very specifically located to avoid misidentification. If
          the speaker name is not whitelisted, skip and alert.
       b. Preprocess the transcript.
       c. Append the transcript to the transcript file of the speaker
          for this video_id. If it is not found, create it. 
  """
  video_id = video_info[0]
  video_fpath = video_info[1]
  game_name = video_info[2]
  print("\n[INFO] Dataset - V-%d - Processing video id %d for %s skits from file: \"%s\"" % (video_id, video_id, game_name, video_fpath))
  wav_fpath = video_fpath.replace(video_suffix, audio_suffix)
  vad_fpath = video_fpath.replace(video_suffix, vad_suffix)

  # First, to work with the audio, we need to extract a wav file from
  # the video, if it doesn't exist aready.
  if create_wav_file(video_fpath, wav_fpath) is False: return

  # Load the wav into memory - reample it if necessary. 
  wav = load_wav(wav_fpath)

  # Attempt to load the video. 
  print("[DEBUG] Dataset - V-%d - Loading video." % video_id)
  cap = cv2.VideoCapture(video_fpath)
  if cap.isOpened() is False: 
    print("[ERROR] Dataset - V-%d - Error opening video file at %s." % (video_id, video_fpath))
    return 

  # Get video statistics. We really hope this is correct. If not, then
  # we'll error out.
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_fps = int(cap.get(cv2.CAP_PROP_FPS))
  print("[INFO] Dataset - V-%d - Video metadata:" % video_id)
  print("                 Video Length (frames): %d" % video_length)
  print("                 Video FPS: %.4f" % video_fps)
  print("")

  # Sometimes, a glitch can happen where the generated wav is not
  # properly generated. Detect that case here and restart generation.
  approx_video_length_ms = video_length * video_fps
  video_wav_diff = abs(approx_video_length_ms - len(wav))
  video_wav_diff = video_wav_diff/approx_video_length_ms
  print("[INFO] Dataset - V-%d - Video aprox length in ms = %d. Audio length = %d. Diff: %.2f" % (video_id, approx_video_length_ms, len(wav), video_wav_diff))
  if video_wav_diff > 0.30:
    print("[ERROR] Dataset - V-%d - Wav file is too short! Throwing out vad mask and wav and trying again..." % video_id)
    os.remove(wav_fpath)
    os.remove(vad_fpath)
    _process_skit_video(video_info, visualization, multiprocessing)
    return

  # Generate tuples of segments of voice activity, each at most 
  # containing one utterance from one speaker. 
  unbuffered_activity_segments = audio_activity_detection(wav, vad_fpath)
  activity_segments, activity_segment_middles = calculate_activity_frames(unbuffered_activity_segments, video_fps)

  statistics = {
    "potential_utterances":len(activity_segments),
    "successful_utterances": 0,
    "total_dropped":0,
    "total_discrepancies": 0,
    "total_no_speaker": 0,
    "total_blacklisted_speaker": 0,
    "total_no_transcript":0,
    "total_cleaner": 0,
    "speaker_verification_failed":0,
  }
  speaker_blacklist = []
  speaker_indices = {}
  transcripts = {}

  # Loop through the video. 
  stop_video = False
  stop_video_frame = None # For debug output only. 
  activity_index = 0
  # Add an additional loop to ensure we reach the end of the video.
  # If we don't, the metadata count is off, for some reason. 
  frames_to_process = video_length +1
  while cap.isOpened() and stop_video is False:

    # Information to propagate to next frame.
    prev_transcript = None
    prev_speaker = None
    prev_drop_utterance = False
    prev_start = None
    cleaner = Cleaner(game_name, multiprocessing)

    # Do not show tqdm progress bar for multiprocessing. 
    if multiprocessing is True:
      enumeration = range(0, frames_to_process)
    else:
      enumeration = tqdm(range(0, frames_to_process), desc="Video Frames Processed", total=frames_to_process)
    
    for frame_num in enumeration:
      if stop_video is False:
        # We read in every single frame to be absolutely sure that we
        # are not missing any frames with audio activity. 
        # Read the next frame of the video. 
        ret, frame = cap.read() 

        if ret:
          # Get current ms since start. If we're >= a middle frame for
          # our next segment, process this frame. 
          current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
          if activity_index < len(activity_segment_middles) and current_ms >= activity_segment_middles[activity_index]:
            # Process the frame if we're not skipping it. 
            frame_ret = process_frame(wav, video_id, frame, frame_num, video_length, 
                                      activity_segments, activity_index, prev_transcript, 
                                      prev_speaker, prev_drop_utterance, prev_start, game_name,
                                      statistics, speaker_blacklist, speaker_indices, cleaner, 
                                      transcripts, visualization, multiprocessing)
            transcripts = frame_ret[0]
            statistics = frame_ret[1]
            speaker_blacklist = frame_ret[2]
            speaker_indices = frame_ret[3]
            last_frame_status = frame_ret[4]
            prev_drop_utterance = frame_ret[5]
            prev_start = frame_ret[6]

            # Depending on the ret, change behavior. 
            if last_frame_status == NEW_UTTERANCE_BAD or last_frame_status == NEW_UTTERANCE_GOOD:
              # If we successfully read the transcript of a new utterance,
              # good or bad, save the information.
              prev_transcript = frame_ret[7]
              prev_speaker = frame_ret[8]
            
            # The next activity frame to look for. 
            activity_index += 1
              
        else:
          # End of video reached. End the loop. 
          stop_video_frame = frame_num
          stop_video = True
          break
    
    # This should hopefully never happen... If it does, the metadata
    # was incorrect and likely the vad mask was incorrect. Die here. 
    assert stop_video is True
    print("[DEBUG] Dataset - V-%d - End of video reached. End frame count: %d" % (video_id, stop_video_frame))
    
  # We've finished processing the video. cleanup.
  cap.release()
  cv2.destroyAllWindows()

  # Execute speaker verification if enabled, going through each folder
  # we possibly generated and verifying all the contents. For files
  # that do not pass speaker verification, we move them to the unclean
  # folder. 
  if speaker_verification_enabled:
    for speaker in speaker_whitelist[game_name]:
      speaker_folder = output_folder + "/" + speaker + "/" + str(video_id)
      if os.path.exists(speaker_folder):
        unclean_files = verify_directory(speaker_folder, video_id)
        statistics["speaker_verification_failed"] += len(unclean_files)
        statistics["total_dropped"] += len(unclean_files)
        statistics["successful_utterances"] -= len(unclean_files)

        # Drop these files from the transcripts file. 
        removed_transcripts = 0
        for removed_file in unclean_files:
          for speaker_name in transcripts:
            speaker_transcripts = transcripts[speaker_name][0]
            removed = False
            for i in range(0, len(speaker_transcripts)):
              if removed_file == speaker_transcripts[i][0] + "." + output_format:
                del speaker_transcripts[i]
                removed_transcripts += 1
                removed = True
                break
            if removed is True: break

        assert(removed_transcripts == len(unclean_files))

  # Write the transcript. 
  for speaker_name in transcripts:
    speaker_transcripts = transcripts[speaker_name][0]
    transcript_path = transcripts[speaker_name][1]

    transcript_file_contents = ""
    for wav_filename, transcript in speaker_transcripts: 
      new_line = "%s %s\n" % (wav_filename, transcript)
      transcript_file_contents += new_line
    
    transcript_file_name = speaker_name+"-"+str(video_id)+".trans.txt"
    f = open(transcript_path + "/" + transcript_file_name, "w")
    f.write(transcript_file_contents)
    print("[INFO] Dataset - V-%d - Wrote %d utterances to transcript file %s." % (video_id, len(speaker_transcripts), transcript_file_name))


  # Print out statistics for user.
  print("[INFO] Dataset - V-%d -Video processing complete. Statistics:" % video_id)
  for key in statistics:
    print("                 %s: %d" % (key, statistics[key]))
  print("")
  print("[INFO] Dataset - V-%d -Blacklisted speakers:" % video_id)
  print(speaker_blacklist)
  print("")

  # Write the same statistics to metadata file. 
  f = open(output_folder + "/" + "video_" + str(video_id) + "_info.txt", "w")

  # Unidecode, as some characters are annoying.
  f.write(unidecode("Video id: %s - %s skits from: \"%s\"\n\n" % (str(video_id), str(game_name), str(video_fpath))))

  f.write("Video metadata:\n")
  f.write("  Video Length (frames): %d\n" % video_length)
  f.write("  Video FPS: %.4f\n" % video_fps)
  f.write("\n")

  f.write("Statistics:\n")
  for key in statistics:
    f.write("  %s: %d\n" % (key, statistics[key]))
  f.write("\n")

  f.write("Blacklisted Speakers:\n")
  for speaker in speaker_blacklist:
    f.write("  %s\n" % speaker)
  f.close()

  # All done with this video.

# When we run, just head right into generation.
# 
# -v = Visualization (default False) 
# -m = Multiprocessing (default True)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", default=False, action="store_true")
  parser.add_argument("-m", default=True, action="store_false")
  args = parser.parse_args()

  visualization = args.v
  multiprocessing = args.m
  extract_tales_skits(visualization, multiprocessing)