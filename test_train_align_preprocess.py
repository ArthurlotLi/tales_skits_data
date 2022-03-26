#
# test_train_align_preprocess.py
#
# Generating alignments from the finalized dataset requires transcripts
# generated in a specific fashion for the Montreal Forced Aligner. 
# This program should be run after the dataset has been generated,
# a test/train split generated, AND the TalesSkits folder copied to
# a new folder called TalesSkitsAligned (with only test and train).

import os
import numpy as np
import shutil

train_location = "./TalesSkitsAligned/train"
test_location = "./TalesSkitsAligned/test"

def process_test_and_train():
  generate_indivdual_transcripts(train_location)
  generate_indivdual_transcripts(test_location)
  move_to_speaker_directories(train_location)
  move_to_speaker_directories(test_location)


def move_to_speaker_directories(directory_fpath):
  """
  For all utterances, move up one subdirectory so that all the
  utterances for a speaker appear in one folder. 
  """
  total_files = 0
  for root, dirs, files in os.walk(directory_fpath):
    for dir in dirs:
      # For each speaker-video directory. 
      full_dir = root + "/" + dir
      dir_files = os.listdir(full_dir)
      dir_files = [full_dir + "/" + file for file in dir_files] 

      files = []
      trans_orig = None
      for dir_file in dir_files:
        if dir_file.endswith(".wav") or dir_file.endswith(".txt") and not dir_file.endswith(".trans.txt"):
          files.append(dir_file)
      
      # Split the wav files into train/test and copy to respective 
      # locations. 
      if len(files) > 0:
        print("[INFO] TestTrainAlign - Processing %s" % full_dir)
        full_dir = full_dir.replace("\\", "/")
        destination_folder = full_dir.rsplit('/', 1)[0]

        for file in files:
          file = file.replace("\\", "/")
          #print("Moving: %s to %s" % (file, destination_folder + "/" + file.rsplit('/', 1)[1]))
          shutil.move(file, destination_folder + "/" + file.rsplit('/', 1)[1])



def generate_indivdual_transcripts(directory_fpath):
  """
  The MFA expects each utterance to be accompanied by a transcript
  with the same name 
  """
  total_files = 0
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
        print("[INFO] TestTrainAlign - Reading transcript %s" % trans_orig)
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

        # Get the files for each item so we can move them. Also append
        # each file's respective transcript. 
        new_transcripts = {}
        for file in wav_files: 
          new_transcripts[file.replace(".wav", ".txt")] = trans_orig_dict[file]

        if len(new_transcripts) > 0:
          for transcript in new_transcripts:
            f_trans_train_path = transcript
            f_trans_train = open(f_trans_train_path, "w")
            f_trans_train.write(new_transcripts[transcript])
            f_trans_train.close()

  print("[INFO] TestTrainAlign - Successfully processed files in %s." % directory_fpath)

if __name__ == "__main__":
  process_test_and_train()