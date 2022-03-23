#
# audio_utils.py
#
# Utilities to support dataset generation related to processing the
# audio of video files. 

from params_data import *

import subprocess
import numpy as np
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
  print("        Wav Length (ms): %d" % wav_length)
  print("        Min Silence (ms): %d" % min_silence)
  print("        Silence Thresh: -%d" % silence_thresh)
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