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

import cv2
import os
from tqdm import tqdm

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

    video_fpath = data_folder + "/" + video_files[video_id-initial_video_id_index]
    print("\n[INFO] Dataset - Processing video id %d for file: \"%s\"" % (video_id, video_fpath))
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
  wav_fpath = video_fpath.replace(video_suffix, audio_suffix)
  vad_fpath = video_fpath.replace(video_suffix, vad_suffix)

  # First, to work with the audio, we need to extract a wav file from
  # the video, if it doesn't exist aready.
  if create_wav_file(video_fpath, wav_fpath) is False: return

  # Load the wav into memory - reample it if necessary. 
  wav = load_wav(wav_fpath)
  
  # Generate a bitmask from the audio - a single bit for each sample
  # via Voice Activity Detection.
  vad_mask = voice_activity_mask(wav, vad_fpath)

  # Attempt to load the video. 
  print("[DEBUG] Dataset - Loading video.")
  cap = cv2.VideoCapture(video_fpath)
  if cap.isOpened() is False: 
    print("[ERROR] Dataset - Error opening video file at %s." % video_fpath)
    return 

  # Get video statistics. We really hope this is correct. If not, then
  # we'll error out but still process to the end of the video. 
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Loop through the video. 
  stop_video = False
  stop_video_frame = None # For debug output only. 
  # Add an additional loop to ensure we reach the end of the video.
  # If we don't, the metadata count is off, for some reason. 
  frames_to_process = length +1
  while cap.isOpened() and stop_video is False:
    for frame_num in tqdm(range(0, frames_to_process), desc="Video Frames Processed", total=frames_to_process):
      if stop_video is False:
        # We read in every single frame to be absolutely sure that we
        # are not missing any frames with audio activity. 
        # Read the next frame of the video. 
        ret, frame = cap.read() 

        if ret:
          if frame_num % frames_to_skip == 0:
            pass
        else:
          # End of video reached. End the loop. 
          stop_video_frame = frame_num
          stop_video = True
          break
  
    if stop_video is False:
      print("[ERROR] Dataset - Video metadata was incorrect!!")
      # Loop again with an indefinite number of frames to process.
      # We won't stop until we actually hit the end of the video. 
      # This should hopefully never happen...
      frames_to_process = int16_max
    else:
      print("[DEBUG] Dataset - End of video reached. End frame count: %d" % stop_video_frame)
  
  # We've finished processing the video. cleanup.
  cap.release()
  cv2.destroyAllWindows()


# When we run, just head right into generation. 
if __name__ == "__main__":
  extract_tales_skits()