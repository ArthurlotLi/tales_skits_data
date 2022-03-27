#
# test_train_align_postprocess.py
#
# Given generated TextGrids for all files, we need to add leading
# silence periods as well as to combine this data with the data of
# our original transcript files to generate final products that can
# mesh seamlessly with multispeaker_synthesis sythesizer 
# preprocessing. Almost there!
#
# Make sure your textgrids are placed in these folders:
#   TalesSkitsAligned/train_textgrids
#   TalesSkitsAligned/test_textgrids
#
# Make sure your original data directory is where it was originally
# generated at TalesSkits.
#
# If a textgrid was not found for a transcript entry, it will be 
# assumed the Montreal Forced Aligner couldn't work with that entry
# and skipped it. That entry will be dropped from our alignments
# file. 
#
# In order to match the preprocessing format of the original 
# alignment files, leading silences must be detected and then 
# written. 
#
# This program will generate a new folder, "final_product" inside
# TalesSkitsAligned that you can use as a drag-and-drop addition
# to the original test train directory.

from data_params import *

import os
import numpy as np
import shutil
from pydub import AudioSegment, silence

train_location = "./TalesSkits/train"
test_location = "./TalesSkits/test"
train_textgrids_location = "./TalesSkitsAligned/train_textgrids"
test_textgrids_location = "./TalesSkitsAligned/test_textgrids"
output_directory = "./TalesSkitsAligned/final_product"

def postprocess_test_and_train():
  process_textgrids_and_transcript(train_location, train_textgrids_location)
  process_textgrids_and_transcript(test_location, test_textgrids_location)

def process_textgrids_and_transcript(directory_fpath, textgrids_fpath):
  """
  Go through and get the transcripts for every subdirectory in
  the original directory structure. 
  """
  total_files = 0
  dropped_files = []
  for root, dirs, files in os.walk(directory_fpath):
    for dir in dirs:
      # For each speaker-video directory. 
      full_dir = root + "/" + dir
      dir_files = os.listdir(full_dir)
      dir_files = [full_dir + "/" + file for file in dir_files] 

      wav_files = []
      trans_orig = None
      for dir_file in dir_files:
        if dir_file.endswith(".wav"):
          dir_file = dir_file.replace("\\", "/")
          wav_files.append(dir_file)
          total_files += 1
        if dir_file.endswith(".trans.txt"):
          dir_file = dir_file.replace("\\", "/")
          trans_orig = dir_file
      
      if len(wav_files) > 0:
        print("[INFO] TestTrainAlign - Processing %s" % full_dir)

        # Manage the transcript. We expect the transcript to be in the
        # same directory and that it contains a single line for every
        # wav file in the directory. 
        assert trans_orig is not None
        #print("[INFO] TestTrainAlign - Reading transcript %s" % trans_orig)
        f_trans_orig = open(trans_orig)
        trans_orig_lines = f_trans_orig.read().split("\n")
        # Remove any entries that are just empty. 
        for i in range(len(trans_orig_lines)-1,-1,-1):
          line = trans_orig_lines[i]
          if line == "" or line == "\n":
            del trans_orig_lines[i]
        assert len(trans_orig_lines) == len(wav_files)

        # For each line in the original transcript, create a map that 
        # matches the wav file. This expects the standardized transcript
        # format, 
        # EX: ZAVEID-15-0005 HOW SHEPHERDLY OF HIM
        #
        # The key should precisely match the full wav file path.
        trans_orig_dict = {}
        for line in trans_orig_lines:
          split_line = line.split(" ", 1)
          wav_filename = split_line[0]
          transcript = split_line[1]
          new_key = full_dir + "/" + wav_filename + ".wav"
          new_key = new_key.replace("\\", "/")
          trans_orig_dict[new_key] = transcript

        # We now need to generate the alignment file. This will also
        # include detecting and adding leading silences to match with 
        # existing preprocessing algorithms. 
        #
        # We will attempt to find the corresponding textgrid directory
        # for this speaker and find all the corresponding TextGrid 
        # entries for each line in the transcript. If a file is 
        # missing, we assume the MFA was unable to align it properly.
        #
        # Ex) train_textgrids/ZAVEID
        full_dir = full_dir.replace("\\", "/")
        speaker_name = full_dir.rsplit('/', 1)[0].rsplit('/', 1)[1]
        speaker_textgrids_location = textgrids_fpath + "/" + speaker_name

        speaker_textgrids = os.listdir(speaker_textgrids_location)
        textgrids = {}
        for file in speaker_textgrids:
          if file.endswith(".TextGrid"):
            textgrid_location = speaker_textgrids_location + "/" + file
            textgrids[full_dir + "/" + file.replace(".TextGrid", ".wav")] = textgrid_location
        
        # We now have two dictionaries - one with transcript lines and
        # one with textgrid locations, both indexed by original wav
        # locations. Determine what's missing. 
        applicable_textgrids = {}
        for wav_location in trans_orig_dict:
          if wav_location not in textgrids:
            dropped_files.append(wav_location.rsplit('/', 1)[1])
          else:
            applicable_textgrids[wav_location] = textgrids[wav_location]
        
        # For each textgrid file, read in the contents, process the wav
        # and add leading silence information. Append to contents of our
        # brand new alignments file in the expected format:
        #
        # Ex) 84-121123-0000 ",GO,,DO,YOU,HEAR," "0.490,0.890,1.270,1.380,1.490,1.890,2.09" 
        align_file_contents = ""
        for wav_location in applicable_textgrids:
          f = open(applicable_textgrids[wav_location])
          textgrid_content = f.read()

          individual_tier = textgrid_content.split("item [1]:")[1].split("item [2]:")[0]
          num_intervals = int(individual_tier.split("intervals: size = ")[1].split("\n")[0])

          assert num_intervals > 0

          # Gather all interval min, max, and text.
          intervals = []
          for i in range(1, num_intervals+1):
            interval_contents = individual_tier.split("intervals [%d]:" % i)[1]
            interval_xmin = float(interval_contents.split("xmin = ", 1)[1].split("\n")[0])
            interval_xmax = float(interval_contents.split("xmax = ", 1)[1].split("\n")[0])
            interval_text = interval_contents.split("text = \"", 1)[1].split("\" \n")[0].strip()
            intervals.append((interval_xmin, interval_xmax, interval_text))
          
          assert len(intervals) > 0

          # Ensure the last tuple is of silence. 
          last_tuple = intervals[len(intervals) - 1]
          assert last_tuple[2] == ""

          # Now with all intervals in tuples, let's manipulate the first
          # one to include silence in the beginning. This means loading 
          # the original wav for this utterance and getting the timestamp
          # of nonsilence. 
          #
          # We will use the same parameters that were used for the audio
          # activity mask generation - particularly the nonsilence buffer.
          # This buffer means that reliably there should be a _ ms buffer
          # of silence for this utterance before audio began. 
          first_tuple = intervals[0]
          if first_tuple[2] != "":

            # Load the wav and detect silence tuples.
            wav = AudioSegment.from_wav(wav_location)
            dBFS=wav.dBFS
            silence_tuples = silence.detect_silence(wav, min_silence_len=nonsilence_buffer_ms, silence_thresh=dBFS-silence_thresh)

            # Get the first silence tuple and use it's ending as the xmax
            # of our new first silence tuple. 
            if len(silence_tuples) == 0 or silence_tuples[0][1]/1000 >= first_tuple[1]:
              first_silence_end = 0
            else:
              first_silence_end = silence_tuples[0][1]/1000

            zero_tuple = (0, first_silence_end, "")
            new_first_tuple = (first_silence_end, first_tuple[1], first_tuple[2])

            del intervals[0]
            intervals = [zero_tuple, new_first_tuple] + intervals

          # Let's generate the new final line for this file. Start with the
          # wav name.
          align_line = wav_location.rsplit('/', 1)[1].replace(".wav", "") + " "

          # Text section.
          align_line += "\""
          text = []
          for interval_tuple in intervals:
            text.append(interval_tuple[2])
          align_line += ",".join(text).upper()
          align_line += "\""

          # Intervals. 
          align_line += " \""
          xmaxs = []
          for interval_tuple in intervals:
            xmaxs.append(str(interval_tuple[1]))
          align_line += ",".join(xmaxs).upper()
          align_line += "\"\n"

          # Append this to the final file. 
          align_file_contents += align_line

        # Write our alignment file. 
        full_dir = full_dir.replace("\\", "/")
        align_location = full_dir.replace(directory_fpath.rsplit("/",1)[0], output_directory)
        speaker_name = full_dir.rsplit('/', 1)[0].rsplit('/', 1)[1]
        video_name = full_dir.rsplit('/', 1)[1]
        align_name = speaker_name + "-" + video_name + ".alignment.txt"
        os.makedirs(align_location, exist_ok=True)
        f_align = open(align_location + "/" + align_name, "w")
        f_align.write(align_file_contents)
        f_align.close()

  print("[INFO] TestTrainAlign - Successfully processed files in %s." % directory_fpath)
  print("                        Dropped: %d" % len(dropped_files))

  os.makedirs(output_directory, exist_ok=True)
  f = open(output_directory + "/" + directory_fpath.split("/", 3)[2] + "_dropped.txt", "w")
  f.write("Dropped: %d\n\n" % len(dropped_files))
  for file in dropped_files:
    f.write("  %s\n" % file)


if __name__ == "__main__":
  postprocess_test_and_train()