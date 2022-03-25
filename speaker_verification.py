#
# speaker_verification.py
#
# "postprocessing" step that verifies that all of the wav files present
# in a speaker's video directory have an acceptable amount of deviation
# from the others, measured by the l2 norm of the difference vector of
# the embeddings.
#
# Requires that the multispeaker_synthesis directory is in the same
# parent directory one level up from this one. Also of course requires
# that the dependencies for multispeaker synthesis have been installed
# (pytorch, etc).

from params_data import *

import os
import sys

sys.path.append(multispeaker_synthesis_fpath)
from production_speaker_verification import *

def verify_directory(wavs_fpath, video_id):
  """
  Given a directory to verify, execute speaker verification and move
  all wavs deemed unclean to the specified directory.

  Returns number of files deemed unclean. 
  """
  reported_files = verify_singular_directory(wavs_fpath, speaker_encoder_fpath, 
                                              speaker_verification_l2_tolerance)
  f = open(unclean_folder + "/" + "verification_results.txt", "a")
  for combined_norms, filename in reported_files:
    old_path = wavs_fpath + "/" + filename
    new_path = unclean_folder + "/" + filename
    if not os.path.exists(unclean_folder): os.makedirs(unclean_folder, exist_ok=True)
    os.replace(old_path, new_path)
    f.write("%.2f - %s\n" % (combined_norms, filename))
  f.close()
  
  print("[INFO] Speaker Verification - V-%d - Deemed %d files as unclean." % (video_id, len(reported_files)))
  return(len(reported_files))