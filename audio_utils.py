#
# audio_utils.py
#
# Utilities to support dataset generation related to processing the
# audio of video files. 

from params_data import *

import struct
import webrtcvad
import subprocess
import numpy as np
from tqdm import tqdm
import os
from pydub import AudioSegment, silence


def create_wav_file(video_fpath, wav_fpath):
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
  wav_creation_command = "ffmpeg -i \"%s\" -ab %s -ac %d -ar %d \"%s\"" % (video_fpath, bit_rate, audio_channels, sample_rate, wav_fpath) 
  print("[INFO] Dataset - Executing command: %s" % wav_creation_command)

  wav_process = subprocess.Popen(wav_creation_command)
  wav_process.wait()

  if os.path.exists(wav_fpath):
    print("[INFO] Dataset - Successfully created wav: %s" % wav_fpath)
    return True

  print("[ERROR] Dataset - Failed to create wav %s!" % wav_fpath)
  return False


def load_wav(wav_fpath):
  """
  Loads and resamples wav if necessary. Returns the wav. 
  Asserts the wav is of the corrct sampling rate.
  """
  # Attempt to load the wav into memory.  
  print("[DEBUG] Dataset - Loading wav into memory.")
  wav = AudioSegment.from_wav(wav_fpath)
  wav_rate = wav.frame_rate
  assert(wav_rate == sample_rate)

  return wav


def audio_activity_detection(wav, vad_fpath):
  """
  Detects ms timestamps of periods of nonsilence in the video. These
  windows should be small enough to virtually guarantee that the 
  maximum number of utterances in a period is at most 1. For each
  tuple, get the (roughly) middle timestamp - this will be used to
  get a reliable subtitle frame correlated to that utterance (or
  partial utterance).
  """
  # Returns milliseconds. 
  wav_length = len(wav)

  # Check if the VAD mask has been generated already. If so, just load
  # it. 
  if os.path.exists(vad_fpath):
    print("[INFO] Dataset - Loading existing VAD mask at: %s" % vad_fpath)
    loaded_mask = np.load(vad_fpath)
    print("[INFO] Dataset - Loaded existing VAD mask with length %d." % len(loaded_mask))
    return loaded_mask

  print("[DEBUG] Dataset - Volume Activity Mask Parameters:")
  print("    Wav Length (ms): %d" % wav_length)
  print("    Min Silence (ms): %d" % min_silence)
  print("    Silence Thresh: -%d" % silence_thresh)
  print("")

  print("[INFO] Dataset - Detecting nonsilence in wav...")
  # Use dBFS (Decibels relative to full scale) for (far) better 
  # results. 
  dBFS=wav.dBFS
  activity_tuples = silence.detect_nonsilent(wav, min_silence_len=min_silence, silence_thresh=dBFS-silence_thresh)

  print("[INFO] Dataset - Detected %d periods of nonsilence." % len(activity_tuples))

  # Write to file. 
  print("[INFO] Dataset - Writing to file: %s" % vad_fpath)
  np.save(vad_fpath, activity_tuples)

  # We're done here. Return the audio mask.
  return activity_tuples


def voice_activity_mask(wav, vad_fpath):
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
  samples_per_window = (vad_window_length * sample_rate) // 1000
  print("[DEBUG] Dataset - VAD Activity Mask Parameters:")
  print("    Wav Length: %d" % wav_length)
  print("    VAD Window Length: %d" % vad_window_length)
  print("    VAD Moving Avg Enabled: %s" % ("yes" if vad_use_moving_average else "no"))
  print("    VAD Moving Avg Window: %d" % vad_moving_average_width)
  print("    Samples Per Window: %d" % samples_per_window)
  print("    Mask fpath: \"%s\"" % vad_fpath)
  print("")

  # Convert the waveform into a tractable 16-bit mono PCM using
  # struct.
  print("[INFO] Dataset - Converting wav to a more tractable 16-bit mono PCM wav.")
  pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

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
                                     sample_rate=sample_rate))

  # Account for the trailing samples that we don't have enough to
  # execute VAD on as window_start points. Use them as window_end
  # points instead. 
  trailing_samples_count = 0
  for window_end in range(wav_length - samples_per_window, wav_length):
    trailing_samples_count += 1
    window_start = window_end - samples_per_window
    voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                     sample_rate=sample_rate))
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
  
  if vad_use_moving_average:
    print("[INFO] Dataset - Smoothing VAD mask with moving average.")
    # Apply the moving average function. Use audio_params for the width.
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    # Renormalize back into boolean flags from our moving average.
    audio_mask = np.round(audio_mask).astype(np.bool)
  else:
    audio_mask = voice_flags

  print("[INFO] Dataset - Audio mask generated. Length: %d" % len(audio_mask))

  # Write to file. 
  print("[INFO] Dataset - Writing audio mask to file: %s" % vad_fpath)
  np.save(vad_fpath, audio_mask)

  # We're done here. Return the audio mask.
  return audio_mask
