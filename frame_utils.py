#
# frame_utils.py
#
# Visual preprocessing functions to aid OCR.

from params_data import *

import cv2
import numpy as np

def preprocess_frame(frame,game_name, subtitles = False):
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


  kernel = np.array([[0, -1, 0],
                     [-1, 5,-1],
                     [0, -1, 0]])

  if game_name == "xillia 1" or game_name == "xillia 2":
    # Convert the frames from BGR to Greyscale. 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize image to make it thin. This is surprisingly effective at
    # reducing variability. 
    frame = cv2.resize(frame, None, fx=1.0, fy=1.6, interpolation=cv2.INTER_CUBIC)
    #frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
  
    # Apply contrast to the image so we can really REALLY read stuff.
    frame = cv2.convertScaleAbs(frame, alpha=2.7, beta=0)

    # Blur
    #frame = cv2.blur(frame,(2,2))

    # Gamma Correction
    frame = adjust_gamma(frame, gamma=0.1)

    # Sharpen
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

    # Extra preprocessing step - remove the icon in the bottom right
    # hand side of the scren.
    if subtitles:
      frame_shape = frame.shape
      y_tot = frame_shape[0]
      x_tot = frame_shape[1]

      x1 = int(0.88 * x_tot)
      y1 = int(0.55 * y_tot)
      x2 = x_tot - int(0.08*x_tot)
      y2 = y_tot - int(0.15*y_tot)

      frame[y1:y2, x1:x2] = (0)

  elif game_name == "berseria" or game_name == "zestiria":
    # Convert the frames from BGR to RGB (better contrast for this font). 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #frame = cv2.resize(frame, None, fx=1.3, fy=1.0, interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(frame, None, fx=0.6, fy=1.0, interpolation=cv2.INTER_AREA)
  
    # Apply contrast to the image so we can really REALLY read stuff.
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=0)

    # Blur
    #frame = cv2.blur(frame,(2,2))

    # Gamma Correction
    frame = adjust_gamma(frame, gamma=0.8)

    # Sharpen
    #frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
  elif game_name == "vesperia":
    # Convert the frames from BGR to Greyscale. 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize image to make it thin. This is surprisingly effective at
    # reducing variability. 
    frame = cv2.resize(frame, None, fx=1.0, fy=1.6, interpolation=cv2.INTER_CUBIC)
    #frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
  
    # Apply contrast to the image so we can really REALLY read stuff.
    frame = cv2.convertScaleAbs(frame, alpha=2.7, beta=0)

    # Blur
    #frame = cv2.blur(frame,(2,2))

    # Gamma Correction
    frame = adjust_gamma(frame, gamma=0.5)

    # Sharpen
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

  else:
    print("[ERROR] Dataset - preprocess_frame received an unknown game name! %s" % game_name)
    assert(False)

  return frame

def get_frame_region_of_interest(frame, game_name):
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

def get_frame_speaker_region(frame, game_name):
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

def debug_frame_view(frame, text_line_1 = None, text_line_2 = None, 
                      text_line_3 = None):
  """
  Debugging visualization tool to show what's going on at a given 
  frame. Can visualize text that you want - up to three optional
  lines. Blocks the main thread indefinitely. 

  Press "q" to advance. 
  """
  frame = frame.copy()
  if text_line_1 is not None:
    cv2.putText(frame, text_line_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  if text_line_2 is not None:
    cv2.putText(frame, text_line_2, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  if text_line_3 is not None:
    cv2.putText(frame, text_line_3, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
  cv2.imshow("TalesSkits Dataset Generation | Debug Frame | Press q to continue", frame)
  while(True):
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break


def calculate_activity_frames(activity_segments, fps):
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