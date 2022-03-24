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

from statistics import variance
from tkinter import FALSE
from params_data import *
from audio_utils import *

import pytesseract
import cv2
import os
from tqdm import tqdm
from difflib import SequenceMatcher

# Enums to make behavior clearer.
NO_AUDIO_ACTIVITY = 1 # VAD says this frame has no activity. Move on. ]
NO_SPEAKER_FOUND = 2
NO_TEXT_FOUND = 3
SAME_UTTERANCE = 4 # VAD says this frame is the same as the current utterance. Move on.
DROP_UTTERANCE = 5
NEW_UTTERANCE_BAD = 6 # A valid utterance, but not accepted transcript. (unknown speaker, bad text)
NEW_UTTERANCE_GOOD = 7 # A new utterance. 

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
  print("                 Video Length (frames): %d" % video_length)
  print("                 Video FPS: %d" % video_fps)
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

    # Information to propagate to next frame.
    prev_transcript = None
    prev_speaker = None
    prev_drop_utterance = False
    prev_start = None
    statistics = {
      "successful_utterances": 0,
      "total_dropped":0,
      "total_discrepancies": 0,
      "total_no_speaker": 0,
      "total_no_transcript":0,
    }

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
          if activity_index < len(activity_segment_middles) and current_ms >= activity_segment_middles[activity_index]:
            # Process the frame if we're not skipping it. 
            frame_ret = _process_frame(video_id, frame, frame_num, video_length, 
                                       activity_segments, activity_index, prev_transcript, 
                                       prev_speaker, prev_drop_utterance, prev_start, game_name,
                                       statistics)
            statistics = frame_ret[0]
            last_frame_status = frame_ret[1]
            prev_drop_utterance = frame_ret[2]
            prev_start = frame_ret[3]

            # Depending on the ret, change behavior. 
            if last_frame_status == NEW_UTTERANCE_BAD or last_frame_status == NEW_UTTERANCE_GOOD:
              # If we successfully read the transcript of a new utterance,
              # good or bad, save the information.
              prev_transcript = frame_ret[4]
              prev_speaker = frame_ret[5]
            
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

  print("[INFO] Dataset - Video processing complete. Statistics:")
  for key in statistics:
    print("                 %s: %d" % (key, statistics[key]))
  print("")

def _process_frame(video_id, frame, frame_num, video_length, 
                   activity_segments, activity_index, prev_transcript, 
                   prev_speaker, prev_drop_utterance, prev_start, game_name,
                   statistics):
  """
  Provided information about the current frame, the current frame
  itself, as well as information regarding the "current" utterance,
  process the frame with OCR. Manage and preprocess the result if it
  is a new utterance. 

  Can return in a few ways: 
  - (status, prev_do_not_save, prev_start) -> not a new utterance.
  - (status, prev_do_not_save, prev_start, prev_transcript, prev_speaker) -> new utterance (good/bad)
  """
  # First, preprocess the frames. Get Regions of Interest and then 
  # process each region for the stuff we need for OCR.
  subtitle_roi = _get_frame_region_of_interest(frame, game_name)
  speaker_roi = _get_frame_speaker_region(frame, game_name)
  subtitle_roi_preprocessed = _preprocess_frame(subtitle_roi)
  speaker_roi_preprocessed = _preprocess_frame(speaker_roi)

  # TODO: Delete me.
  #frame = _preprocess_frame(frame)

  new_utterance = None
  drop_current_utterance = False

  # OCR. First, read the speaker name. Preprocess the name so
  # that we can look it up in the game's speaker name whitelist. 
  speaker_name = pytesseract.image_to_string(speaker_roi_preprocessed)
  speaker_name = speaker_name.replace("\n", "").lower()

  # If no speaker is found, this is considered a NEW utterance and
  # the previous utterance (if present) to be complete. 
  if speaker_name is None or speaker_name == "":
    print("\n[WARNING] Dataset - No speaker found!")
    #_debug_frame_view(frame, "WARNING - NO SPEAKER FOUND!")
    speaker_name = ""
    new_utterance = True
    drop_current_utterance = True
    statistics["total_no_speaker"] += 1

  # If the speaker name is different, we don't even need to compare 
  # the transcript - we know it's a new utterance. OCR the trancript.
  # Otherwise if the same speaker, OCR the transcript and compare to
  # previous transcript. If it's the same, same.
  if prev_speaker is None or prev_speaker != speaker_name:
    new_utterance = True

  subtitles = pytesseract.image_to_string(subtitle_roi_preprocessed)

  # Toss out any frames without subtitles. 
  if subtitles is None or subtitles == "":
    # Warn the user. But if we already know this utterance is bad, do
    # not bother the user - we ale already treating this utterance as
    # a new, bad utterance.
    if drop_current_utterance is False:
      print("\n[WARNING] Dataset - No transcript found!")
      #_debug_frame_view(frame, "WARNING - NO TRANSCRIPT FOUND!", "Speaker: %s" % speaker_name)
      subtitles = ""
      new_utterance = True
      drop_current_utterance = True
      statistics["total_no_transcript"] += 1

  if new_utterance is None:
    # Time to compare transcripts. We will specifically parse the % that
    # the current transcript matches the previous. under a certain 
    # threshold, the previous transcript wil be assumed to be equal 
    # to the existing one, as the chances of the same speaker uttering
    # the nearly same (but not different) text directly after uttering
    # it is assumed to be near zero. 
    variance_from_prev_transcript = SequenceMatcher(None, subtitles, prev_transcript).ratio()

    if variance_from_prev_transcript <= subtitle_variance_thresh:
      new_utterance = True
    else:
      new_utterance = False
      # Don't bother check if we already know this utterrance is corrupted. 
      if prev_drop_utterance is False and variance_from_prev_transcript <= 1.0 - subtitle_variance_acceptable_thresh:
        print("\n[WARNING] Dataset - Potential discrepancy found! Prev start: %d" % prev_start)
        print("                    Prev (%s): \"%s\"" % (prev_speaker, prev_transcript.replace("\n", " ")))
        print("                    Current (%s): \"%s\"" % (speaker_name, subtitles.replace("\n", " ")))
        print("                    Similarity: %.2f" % variance_from_prev_transcript)
        #_debug_frame_view(frame, "WARNING - %.2f MATCH!" % variance_from_prev_transcript, "Prev: \"%s\"" % prev_transcript, "Current: \"%s\"" % subtitles)

        # We don't consider this to be a new utterance if this is the case. 
        # Mark the current utterance as corrupted and continue looking
        # for it, so we know when it ends. 
        drop_current_utterance = True
        statistics["total_discrepancies"] += 1

  if new_utterance is False:
    # Manage the possibility of a corrupted utterance. In this case,
    # (discrepancy with text), turn on drop flag. 
    if drop_current_utterance is True:
      return(statistics, DROP_UTTERANCE, True, prev_start, subtitles, speaker_name)
    else:
      return(statistics, SAME_UTTERANCE, prev_drop_utterance, prev_start)

  # At this point we KNOW this is now a NEW UTTERANCE. Wrap up the 
  # previous utterance. Write the wav to file and append the 
  # transcript to the speaker's transcript file for this video_id. 
  complete_utterance_start = prev_start
  complete_utterance_end = activity_segments[activity_index-1][1]
  complete_utterance_speaker = prev_speaker
  if not prev_drop_utterance or drop_current_utterance:
    complete_utterance_transcript = prev_transcript

    statistics["successful_utterances"] += 1
  else:
    print("\n[DEBUG] Dataset - Dropped utterance (%s) with range: %d - %d" 
      % (complete_utterance_speaker, complete_utterance_start, complete_utterance_end))
    statistics["total_dropped"] += 1

  # Now, start a new utterance. Make sure the speaker name is 
  # whitelisted for this particular game name. If not, alert user.
  # This is a bad utterance. 


  # If the speaker name is whitelisted, process the transcript with
  # preprocessing to match LibriSpeech format. If the transcript 
  # ends up empty, this is a bad utterance.

  # If everything is good up until this point, we're all set. 
    
  start = activity_segments[activity_index][0]
  #end = activity_segments[activity_index][1]
  #middle = int(((end - start)//2 ) + start)

  #text_line_1, text_line_2, text_line_3 = None, None, None
  #text_line_1 = "New (%s): \"%s\"" % (speaker_name, subtitles)
  #text_line_3 = "AI: %d Start: %d End: %d Middle: %d Frame: %d" % (activity_index,start, end, middle, frame_num)
  #if prev_start is not None and prev_speaker is not None:
    #text_line_2 = "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript)
    #text_line_3 = "Prev Drop: %s | Prev Start: %d | Prev End: %d" % (str(prev_drop_utterance), prev_start, activity_segments[activity_index-1][1])
  #_debug_frame_view(frame, text_line_1=text_line_1, text_line_2 = text_line_2, text_line_3 = text_line_3)

  return(statistics, NEW_UTTERANCE_GOOD, drop_current_utterance, start, subtitles, speaker_name)

def _debug_frame_view(frame, text_line_1 = None, text_line_2 = None, 
                      text_line_3 = None):
  """
  Debugging visualization tool to show what's going on at a given 
  frame. Can visualize text that you want - up to three optional
  lines. Blocks the main thread indefinitely. 

  Press "q" to advance. 
  """
  if text_line_1 is not None:
    cv2.putText(frame, text_line_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  if text_line_2 is not None:
    cv2.putText(frame, text_line_2, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  if text_line_3 is not None:
    cv2.putText(frame, text_line_3, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  cv2.imshow("TEST", frame)
  while(True):
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

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


def _preprocess_frame(frame):
  """
  Given a frame, prerocess the frame in preparation for OCR.

  As usual, greyscale the image. Add contrast.

  Note: Does not do ROI processing. See below for those. 
  Return the processed frame. 
  """
  def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

  # Convert the frames from BGR to Greyscale. 
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  alpha = 2.7 # Contrast control (1.0-3.0)
  beta = 0 # Brightness control (0-100)
  kernel = np.array([[0, -1, 0],
                     [-1, 5,-1],
                     [0, -1, 0]])
  resize_x = 2
  resize_y = 2
  gamma = 0.1

  # Resize image to make it thin. This is surprisingly effective at
  # reducing variability. 
  frame = cv2.resize(frame, None, fx=1.0, fy=1.6, interpolation=cv2.INTER_CUBIC)
  #frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
 
  # Apply contrast to the image so we can really REALLY read stuff.
  frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

  # Blur
  #frame = cv2.blur(frame,(2,2))

  # Gamma Correction
  frame = adjust_gamma(frame, gamma=gamma)

  # Sharpen
  frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

  # Blur
  #frame = cv2.blur(frame,(3,3))

  # Sharpen
  #frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

  return frame

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