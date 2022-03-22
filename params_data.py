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
frames_to_skip = 3
video_suffix = ".mp4"

# Audio configruation
sample_rate = 16000 # In Hz
bit_rate = "160k" # kbps
audio_channels = 1
audio_suffix = ".wav"
vad_moving_average_width = 8 # Samples. Larger values - less smooth.
vad_window_length = 10 # Milliseconds (10, 20, or 30) - Granularity of the VAD operation
vad_suffix = ".vad_mask.npy"

# Region of interest for videos. Where the subtitles + name
# should appear. Made in terms of percentage of x and y as
# we have variable resolution data. 
subtitle_roi_by_game = {
  "berseria" : {
    "subtitle_roi_x1": .15, # % total resolution from left.
    "subtitle_roi_y1": .75, # % total resolution from top.
    "subtitle_roi_x2": .15, # % total resolution from right.
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