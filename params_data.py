#
# params_data.py
#
# Configurable parameters for dataset generation.

data_folder = "./data_test"
output_folder = "./TalesSkits"

# Used in trim_long_silences - largest number in int16.
int16_max = (2 ** 15) - 1

# Video configuration
initial_video_id_index = 1
# Run OCR every _ frames to check for start/end of utterance
frames_to_skip = 1
video_suffix = ".mp4"

# Audio configruation
sample_rate = 16000 # In Hz
bit_rate = "160k" # kbps
audio_channels = 1
audio_suffix = ".wav"

use_silence_instead_of_vad = True

vad_moving_average_width = 4 # Samples. Larger values - less smooth.
vad_use_moving_average = False
vad_window_length = 30 # Milliseconds (10, 20, or 30) - Granularity of the VAD operation
vad_suffix = ".vad_mask.npy"

min_silence = 10 # In milliseconds. Lower values are more accurate, but risk excessive OCR processing.
silence_thresh = 42 # Subtracted from dBFS. Should be as high (low) as possible, since this is CLEAN data. 
nonsilence_buffer_ms = 5 # A buffer into silence for each segment of activity.
min_length_of_non_silence = 0 # Should avoid artifacts. NOTE: Disabled. Better safe than sorry. 

# Region of interest for videos. Where the subtitles + name
# should appear. Made in terms of percentage of x and y as
# we have variable resolution data. 
subtitle_roi_by_game = {
  "berseria" : {
    "subtitle_roi_x1": .15, # % total resolution from left.
    "subtitle_roi_y1": .75, # % total resolution from top.
    "subtitle_roi_x2": .10, # % total resolution from right.
    "subtitle_roi_y2": .01, # % total resolution from bottom. 
  },
  "zestiria" : {
    "subtitle_roi_x1": .08, # % total resolution from left.
    "subtitle_roi_y1": .78, # % total resolution from top.
    "subtitle_roi_x2": .08, # % total resolution from right.
    "subtitle_roi_y2": .00, # % total resolution from bottom. 
  },
  "xillia 1" : {
    "subtitle_roi_x1": .15, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .15, # % total resolution from right.
    "subtitle_roi_y2": .04, # % total resolution from bottom. 
  },
  "xillia 2" : {
    "subtitle_roi_x1": .15, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .15, # % total resolution from right.
    "subtitle_roi_y2": .01, # % total resolution from bottom. 
  },
  "vesperia" : {
    "subtitle_roi_x1": .05, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .03, # % total resolution from right.
    "subtitle_roi_y2": .01, # % total resolution from bottom. 
  },
}

# Region where the speaker should appear. Made in terms of 
# percentage of x and y aswe have variable resolution data. 
speaker_roi_by_game = {
  "berseria" : {
    "subtitle_roi_x1": .15, # % total resolution from left.
    "subtitle_roi_y1": .75, # % total resolution from top.
    "subtitle_roi_x2": .50, # % total resolution from right.
    "subtitle_roi_y2": .175, # % total resolution from bottom. 
  },
  "zestiria" : {
    "subtitle_roi_x1": .08, # % total resolution from left.
    "subtitle_roi_y1": .78, # % total resolution from top.
    "subtitle_roi_x2": .50, # % total resolution from right.
    "subtitle_roi_y2": .176, # % total resolution from bottom. 
  },
  "xillia 1" : {
    "subtitle_roi_x1": .232, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .50, # % total resolution from right.
    "subtitle_roi_y2": .212, # % total resolution from bottom. 
  },
  "xillia 2" : {
    "subtitle_roi_x1": .232, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .50, # % total resolution from right.
    "subtitle_roi_y2": .212, # % total resolution from bottom. 
  },
  "vesperia" : {
    "subtitle_roi_x1": .05, # % total resolution from left.
    "subtitle_roi_y1": .73, # % total resolution from top.
    "subtitle_roi_x2": .76, # % total resolution from right.
    "subtitle_roi_y2": .10, # % total resolution from bottom.
  },
}