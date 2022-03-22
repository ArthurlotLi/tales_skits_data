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
#      during non-voice SFX instances. 
#
# Ensure that audio configuration hyperparameters match those of the
# project that you want to apply this dataset to. 

import cv2
import os
from tqdm import tqdm
import subprocess
import numpy as np
import librosa
import struct
import webrtcvad

_data_folder = "./data"
# Used in trim_long_silences - largest number in int16.
_int16_max = (2 ** 15) - 1

# Video configuration
_initial_video_id_index = 1
_frames_to_skip = 30
_video_suffix = ".mp4"

# Audio configruation
_sample_rate = 16000 # In Hz
_bit_rate = "160k" # kbps
_audio_channels = 1
_audio_suffix = ".wav"
_vad_moving_average_width = 8 # Samples. Larger values - less smooth.
_vad_window_length = 10 # Milliseconds (10, 20, or 30) - Granularity of the VAD operation


def extract_tales_skits():
  """
  Principal function extracting utterances by speaker with transcripts
  from Tales of video game skit recordings. 
  """
  if not os.path.exists(_data_folder):
    print("[ERROR] Dataset - Unable to find %s." % _data_folder)
  
  # Process every single video, giving each video it's own identifier.
  data_contents = os.listdir(_data_folder)
  data_count = len(data_contents)
  print("[INFO] Dataset - found %d files in folder %s." % (data_count,_data_folder))
  for video_id in range(_initial_video_id_index, data_count+_initial_video_id_index):
    video_fpath = _data_folder + "/" + data_contents[video_id-_initial_video_id_index]
    print("\n[INFO] Dataset - Processing video id %d: %s" % (video_id, video_fpath))
    _process_skit_video(video_id, video_fpath)

def _process_skit_video(video_id, video_fpath):
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

  # First, to work with the audio, we need to extract a wav file from
  # the video, if it doesn't exist aready.
  wav_fpath = video_fpath.replace(_video_suffix, _audio_suffix)
  vad_fpath = video_fpath.replace(_video_suffix, ".npy")
  if _create_wav_file(video_fpath, wav_fpath) is False: return

  # Attempt to load the wav into memory. 
  print("[DEBUG] Dataset - Loading wav into memory.")
  wav, source_sr = librosa.load(wav_fpath, sr=None)

  # Resampling to match the expected sampling rate in audio_params if
  # necessary.
  if source_sr is not None and source_sr != _sample_rate:
    print("[INFO] Dataset - Source sample rate %d does not match set sample rate %d! Resampling." % (source_sr, _sample_rate))
    wav = librosa.resample(wav, orig_sr=source_sr, target_sr=_sample_rate)
  
  # Generate a bitmask from the audio - a single bit for each sample
  # courtesy of Voice Activity Detection.
  vad_mask = _voice_activity_mask(wav, vad_fpath)

  # Attempt to load the video. 
  print("[DEBUG] Dataset - Loading video.")
  cap = cv2.VideoCapture(video_fpath)
  if cap.isOpened() is False: 
    print("[ERROR] Dataset - Error opening video file at %s." % video_fpath)
    return 

  # Get video statistics. 
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = cap.get(cv2.CAP_PROP_FPS)

  # Loop through the video. 
  stop_video = False
  # Add an additional loop to ensure we reach the end of the video.
  # If we don't, the metadata count is off, for some reason. 
  frames_to_process = length +1
  while cap.isOpened() and stop_video is False:
    for frame_num in tqdm(range(0, frames_to_process), desc="Video Frames Processed", total=frames_to_process):
      # We read in every single frame to be absolutely sure that we
      # are not missing any frames with audio activity. 

      # Read the next frame of the video. 
      ret, frame = cap.read() 

      if ret:
        if frame_num % _frames_to_skip == 0:
          pass
      else:
        # End of video reached. End the loop. 
        print("[DEBUG] Dataset - End of video reached.")
        stop_video = True
        break
  
    if stop_video is False:
      print("[ERROR] Dataset - Video metadata was incorrect!!")
      # Loop again with an indefinite number of frames to process.
      # We won't stop until we actually hit the end of the video. 
      # This should hopefully never happen...
      frames_to_process = _int16_max
  
  # We've finished processing the video. cleanup.
  cap.release()
  cv2.destroyAllWindows()

def _create_wav_file(video_fpath, wav_fpath):
  """
  Given the fpath of the source video, extract a wav file with
  ffmpeg. Expects ffmpeg to exist on the machine. Returns true or false
  depending on whether the wav file exists. 
  """
  # If the wav file exists, just end. 
  if os.path.exists(wav_fpath): 
    print("[INFO] Dataset - Using existing wav: %s" % wav_fpath)
    return True

  # If the wav doesn't exist, create it. 
  wav_creation_command = "ffmpeg -i \"%s\" -ab %s -ac %d -ar %d \"%s\"" % (video_fpath, _bit_rate, _audio_channels, _sample_rate, wav_fpath) 
  print("[INFO] Dataset - Executing command: %s" % wav_creation_command)

  wav_process = subprocess.Popen(wav_creation_command)
  wav_process.wait()

  if os.path.exists(wav_fpath):
    print("[INFO] Dataset - Successfully created wav: %s" % wav_fpath)
    return True

  print("[ERROR] Dataset - Failed to create wav %s!" % wav_fpath)
  return False

def _voice_activity_mask(wav, vad_fpath):
  """
  Given a read in wav of video, generate a VAD mask that indicates
  where audio activity is occuring in the video. This mask needs to
  be precisely lined up with the frames of the video.

  We'll use a small moving average to even out small spikes. 

  While this method does result in many ultimately unused calculations,
  it does result in the most accurate frame-to-audio mapping. 
  """
  wav_length = len(wav)

  # Check if the VAD mask has been generated already. If so, just load
  # it. 
  if os.path.exists(vad_fpath):
    print("[INFO] Dataset - Loading existing VAD mask at: %s" % vad_fpath)
    loaded_mask = np.load(vad_fpath)
    loaded_mask_length = len(loaded_mask)
    print("[INFO] Dataset - Loaded existing VAD mask with length %d." % loaded_mask_length)
    if loaded_mask_length != wav_length:
      print("[WARNING] Dataset - Loaded existing VAD mask length %d does NOT match wav length %d. Overwriting..." % (loaded_mask_length, wav_length))
    else:
      return loaded_mask

  # We want a mask for EVERY SINGLE SAMPLE in this wav file. This
  # allows us to reference the bitmask in the same context as the
  # wav samples.
  samples_per_window = (_vad_window_length * _sample_rate) // 1000
  print("[DEBUG] Dataset - VAD Activity Mask Parameters:")
  print("    Wav Length: %d" % wav_length)
  print("    VAD Window Length: %d" % _vad_window_length)
  print("    Samples Per Window: %d" % samples_per_window)
  print("    VAD fpath: \"%s\"" % vad_fpath)
  print("")

  # Convert the waveform into a tractable 16-bit mono PCM using
  # struct.
  print("[INFO] Dataset - Converting wav to a more tractable 16-bit mono PCM wav.")
  pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * _int16_max)).astype(np.int16))

  # Now we grab the binary flags by executing voice activation
  # detection using webrtcvad. Iterate through the wav and, for
  # each window, get a flag saying whether it is active or not.
  # We iterate through EVERY single step. Skip the last few parts
  # of the wav that we will not have enough frames to process. 
  voice_flags = []
  vad = webrtcvad.Vad(mode=3)
  print("[INFO] Dataset - Processing VAD mask.")
  for window_start in tqdm(range(0, wav_length - samples_per_window), desc="VAD Samples Processed"):
    window_end = window_start + samples_per_window
    voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                     sample_rate=_sample_rate))

  # Account for the trailing samples that we don't have enough to
  # execute VAD on as window_start points. Use them as window_end
  # points instead. 
  trailing_samples_count = 0
  for window_end in range(wav_length - samples_per_window, wav_length):
    trailing_samples_count += 1
    window_start = window_end - samples_per_window
    voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                     sample_rate=_sample_rate))
  print("[INFO] Dataset - Processed %d trailing samples." % trailing_samples_count)
  
  # Given our flags, apply a moving average to remove spikes
  # in the voice_flags array. We'll use a helper function for
  # this - given the array and the width of the moving average,
  # return the array with the moving average applied. 
  def moving_average(array, width):
    # Pad the array on both sides with (w-1//2) length zero vectors.
    # This is to help the moving average on the ends. 
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    # Cumulative sum of elements along a given axis. 
    ret = np.cumsum(array_padded, dtype=float)
    # Clean up the length of the array. 
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width
  
  print("[INFO] Dataset - Smoothing VAD mask with moving average.")
  # Apply the moving average function. Use audio_params for the width.
  audio_mask = moving_average(voice_flags, _vad_moving_average_width)
  # Renormalize back into boolean flags from our moving average.
  audio_mask = np.round(audio_mask).astype(np.bool)

  print("[INFO] Dataset - Audio mask generated. Length: %d" % len(audio_mask))

  # Write to file. 
  print("[INFO] Dataset - Writing audio mask to file: %s" % vad_fpath)
  np.save(vad_fpath, audio_mask)

  # We're done here. Return the audio mask.
  return audio_mask

if __name__ == "__main__":
  extract_tales_skits()