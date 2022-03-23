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

from params_data import *
from audio_utils import *

import math
import cv2
import os
from tqdm import tqdm

# Enums to make behavior clearer.
NO_AUDIO_ACTIVITY = 1 # VAD says this frame has no activity. Move on. ]
NO_TEXT_FOUND = 2
SAME_UTTERANCE = 3 # VAD says this frame is the same as the current utterance. Move on.
NEW_UTTERANCE_BAD = 4 # A valid utterance, but not accepted transcript. (unknown speaker, bad text)
NEW_UTTERANCE_GOOD = 5 # A new utterance. 

def extract_tales_skits():
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
  for video_id in range(initial_video_id_index, data_count+initial_video_id_index):
    
    # TODO: Implement multiprocessing and use batches. 
    video_filename = video_files[video_id-initial_video_id_index]
    video_fpath = data_folder + "/" + video_filename
    game_name = _determine_game_title(video_filename)
    print("\n[INFO] Dataset - Processing video id %d for %s skits from file: \"%s\"" % (video_id, game_name, video_fpath))
    _process_skit_video(video_id, video_fpath, game_name)

def _determine_game_title(filename):
  """
  Given the filename, retrieve the game title code. This will help
  allow us to behave differently depending on the game skit format. 
  """
  lower_filename = filename.lower()
  for game_name in subtitle_roi_by_game:
    if game_name in lower_filename:
      return game_name

def _process_skit_video(video_id, video_fpath, game_name):
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
  wav_fpath = video_fpath.replace(video_suffix, audio_suffix)
  vad_fpath = video_fpath.replace(video_suffix, vad_suffix)

  # First, to work with the audio, we need to extract a wav file from
  # the video, if it doesn't exist aready.
  if create_wav_file(video_fpath, wav_fpath) is False: return

  # Load the wav into memory - reample it if necessary. 
  wav = load_wav(wav_fpath)

  # Attempt to load the video. 
  print("[DEBUG] Dataset - Loading video.")
  cap = cv2.VideoCapture(video_fpath)
  if cap.isOpened() is False: 
    print("[ERROR] Dataset - Error opening video file at %s." % video_fpath)
    return 

  # Get video statistics. We really hope this is correct. If not, then
  # we'll error out.
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  video_fps = int(cap.get(cv2.CAP_PROP_FPS))
  print("[INFO] Dataset - Video metadata:")
  print("       Video Length (frames): %d" % video_length)
  print("       Video FPS: %d" % video_fps)
  print("")

  # Generate tuples of segments of voice activity, each at most 
  # containing one utterance from one speaker. 
  unbuffered_activity_segments = audio_activity_detection(wav, vad_fpath)
  activity_segments, activity_segment_middles = _calculate_activity_frames(unbuffered_activity_segments, video_fps)

  # Loop through the video. 
  stop_video = False
  stop_video_frame = None # For debug output only. 
  activity_index = 0
  # Add an additional loop to ensure we reach the end of the video.
  # If we don't, the metadata count is off, for some reason. 
  frames_to_process = video_length +1
  while cap.isOpened() and stop_video is False:
    prev_transcript = None
    prev_start = None
    last_frame_status = None
    for frame_num in tqdm(range(0, frames_to_process), desc="Video Frames Processed", total=frames_to_process):
      if stop_video is False:
        # We read in every single frame to be absolutely sure that we
        # are not missing any frames with audio activity. 
        # Read the next frame of the video. 
        ret, frame = cap.read() 

        if ret:
          # Get current ms since start. If we're >= a middle frame for
          # our next segment, process this frame. 
          current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
          if current_ms >= activity_segment_middles[activity_index]:
            # Process the frame if we're not skipping it. 
            frame_ret = _process_frame(video_id, frame, frame_num, video_length, 
                                       activity_segments, activity_index, prev_transcript, 
                                       prev_start, last_frame_status, game_name)
            last_frame_status = frame_ret[0]

            # Depending on the ret, change behavior. 
            if last_frame_status == NEW_UTTERANCE_BAD or last_frame_status == NEW_UTTERANCE_GOOD:
              # If we successfully read the transcript of a new utterance,
              # good or bad, save the information.
              prev_transcript = frame_ret[1]
              prev_start = frame_ret[2]
            
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
    print("[DEBUG] Dataset - End of video reached. End frame count: %d" % stop_video_frame)
  
  # We've finished processing the video. cleanup.
  cap.release()
  cv2.destroyAllWindows()

def _process_frame(video_id, frame, frame_num, video_length, 
                   activity_segments, activity_index, prev_transcript, 
                   prev_start, last_frame_status, game_name):
  """
  Provided the video_id, the full-sized cv2 frame extracted from the
  source video, the frame number + total frames of the source video,
  the vad_mask, and current utterance + frame data, process it. 

  Can return in a few ways: 
  - (status,) -> not a new utterance.
  - (status, prev_transcript, prev_start) -> new utterance (good/bad)
  """
  roi_frame = _get_frame_region_of_interest(frame, game_name)
  speaker_frame = _get_frame_speaker_region(frame, game_name)

  # DEBUG ONLY! Visualize the frame and indicate sound activity. 
  start = activity_segments[activity_index][0]
  end = activity_segments[activity_index][1]
  middle = int(((end - start)//2 ) + start)
  debug_tuple = (activity_index,start, end, middle, frame_num)
  #cv2.putText(roi_frame, "AI: %d Start: %d End: %d Middle: %d Frame: %d" % debug_tuple, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow("TEST", speaker_frame)
  while(True):
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  return (NO_AUDIO_ACTIVITY,)

def _calculate_activity_frames(activity_segments, fps):
  """
  Given tuples of nonsilence in the wav, for each tuple, get the
  millisecond timestamp of the (rough) middle for a reliable frame
  showing utterance subtitles. Also add a buffer to each start and 
  end. 

  Return BOTH final activity frames tuples, each item being:
  (start (ms buffered), end (ms buffered), middle_frame)
  AS WELL as a list of middle frames by itself. These two lists
  should be like indexed. 
  """
  activity_segment_middles = []
  new_activity_segments = []
  total_dropped_segments = 0
  for start, end in activity_segments:
    # Preprocessing - if the original length is too short, drop this
    # activity segment. 
    segment_length = end-start
    if segment_length < min_length_of_non_silence:
      total_dropped_segments  += 1
      continue

    middle = ((segment_length)//2 ) + start
    start = start - nonsilence_buffer_ms
    end = end + nonsilence_buffer_ms
    activity_segment_middles.append(middle)
    new_activity_segments.append((start, end))

  print("[DEBUG] Datset - Dropped a total of %d segments with length less than %d ms." % (total_dropped_segments, min_length_of_non_silence))

  # Verify that all samples are NON OVERLAPPING, even with the 
  # buffers in place. 
  print("[DEBUG] Dataset - Verifying buffered activity segments are all non-overlapping.")
  for i in range(0, len(new_activity_segments) - 1):
    first_tuple = new_activity_segments[i]
    second_tuple = new_activity_segments[i + 1]

    if(first_tuple[1] > second_tuple[0]):
      print("[ERROR] Dataset - Activity segments overlap detected!! Tuples: ")
      print(first_tuple)
      print(second_tuple)
      print("Aborting!")
      assert(False)

  return new_activity_segments, activity_segment_middles

def _get_frame_region_of_interest(frame, game_name):
  """
  Given a frame, returns the region of interest. OpenCV stores images
  in numpy arrays, so this is pretty easy. 
  """
  # We expect a tensor (y pixels, x pixels, channels (3 for RGB))
  # Ex) (1080, 1920, 3)
  frame_shape = frame.shape
  y_tot = frame_shape[0]
  x_tot = frame_shape[1]

  x1 = int(subtitle_roi_by_game[game_name]["subtitle_roi_x1"] * x_tot)
  y1 = int(subtitle_roi_by_game[game_name]["subtitle_roi_y1"] * y_tot)
  x2 = x_tot - int(subtitle_roi_by_game[game_name]["subtitle_roi_x2"] * x_tot)
  y2 = y_tot - int(subtitle_roi_by_game[game_name]["subtitle_roi_y2"] * y_tot)

  roi_frame = frame[y1:y2, x1:x2]
  return roi_frame

def _get_frame_speaker_region(frame, game_name):
  """
  Given a frame, returns the speaker name region. OpenCV stores images
  in numpy arrays, so this is pretty easy. 
  """
  # We expect a tensor (y pixels, x pixels, channels (3 for RGB))
  # Ex) (1080, 1920, 3)
  frame_shape = frame.shape
  y_tot = frame_shape[0]
  x_tot = frame_shape[1]

  x1 = int(speaker_roi_by_game[game_name]["subtitle_roi_x1"] * x_tot)
  y1 = int(speaker_roi_by_game[game_name]["subtitle_roi_y1"] * y_tot)
  x2 = x_tot - int(speaker_roi_by_game[game_name]["subtitle_roi_x2"] * x_tot)
  y2 = y_tot - int(speaker_roi_by_game[game_name]["subtitle_roi_y2"] * y_tot)

  roi_frame = frame[y1:y2, x1:x2]
  return roi_frame

# When we run, just head right into generation. 
if __name__ == "__main__":
  extract_tales_skits()