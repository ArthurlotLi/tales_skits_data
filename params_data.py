#
# params_data.py
#
# Configurable parameters for dataset generation.

data_folder = "./data"
output_folder = "./TalesSkits"

# Used in trim_long_silences - largest number in int16.
int16_max = (2 ** 15) - 1

# Video configuration
initial_video_id_index = 1
frames_to_skip = 30
video_suffix = ".mp4"

# Audio configruation
sample_rate = 16000 # In Hz
bit_rate = "160k" # kbps
audio_channels = 1
audio_suffix = ".wav"
vad_moving_average_width = 8 # Samples. Larger values - less smooth.
vad_window_length = 10 # Milliseconds (10, 20, or 30) - Granularity of the VAD operation
vad_suffix = ".vad_mask.npy"