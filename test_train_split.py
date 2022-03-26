#
# test_train_split.py
#
# Simple mechanism to take the contents of the TalesSkits "full" data
# and to and to copy them into test and train folders with a specified
# split. 
#
# Splits by extracting _ % of the utterances in every folder into the
# corresponding location in the train folder. Remaining utterances
# are sent to the corresponding locaiton in the test folder. 
#
# Assumes data is in the present in the directory specified in params
# data. 
#
# Also properly manages the transcripts for every directory, ensuring 
# that the two product directories are provided accurate transcripts
# for their contents. 

from data_params import *

import os
import numpy as np
import shutil

train_percentage = 0.90
train_location = "./TalesSkits/train"
test_location = "./TalesSkits/test"

def execute_split():
  """
  Assuming data is present in the output_folder specified in params
  data, executes the split by copying files. 
  """
  total_files = 0
  train_files = 0
  test_files = 0
  for root, dirs, files in os.walk(output_folder):
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
      
      # Split the wav files into train/test and copy to respective 
      # locations. 
      if len(wav_files) > 0:
        print("[INFO] TestTrainSplit - Processing %s" % full_dir)

        # Manage the transcript. We expect the transcript to be in the
        # same directory and that it contains a single line for every
        # wav file in the directory. 
        assert trans_orig is not None
        print("[INFO] TestTrainSplit - Reading transcript %s" % trans_orig)
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
          new_key = full_dir + "/" + line.split(" ", 1)[0] + ".wav"
          new_key = new_key.replace("\\", "/")
          trans_orig_dict[new_key] = line

        trans_train_content = ""
        trans_test_content = ""

        train, test = np.split(wav_files, [int(len(wav_files)*train_percentage)])
        old_train = train.copy()
        old_test = test.copy()

        # Get the files for each item so we can move them. Also append
        # each file's respective transcript. 
        new_train = []
        new_test = []
        for file in train: 
          trans_train_content += trans_orig_dict[file] + "\n"
          new_train.append(file.replace(output_folder, train_location))
        for file in test: 
          trans_test_content += trans_orig_dict[file] + "\n"
          new_test.append(file.replace(output_folder, test_location))

        # Get the folders for each item, so we can create them. 
        train_path = new_train[0].rsplit('/', 1)[0]
        test_path = new_test[0].rsplit('/', 1)[0]

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # We now have full paths for all of our new files. Copy.
        for i in range(0, len(new_train)):
          shutil.copyfile(old_train[i], new_train[i])
          train_files += 1
        
        for i in range(0, len(new_test)):
          shutil.copyfile(old_test[i], new_test[i])
          test_files += 1
        
        # Finally, write the transcripts. 
        f_trans_train_path = train_path + "/" + trans_orig.rsplit('/', 1)[1]
        f_trans_test_path = test_path + "/" + trans_orig.rsplit('/', 1)[1]
        print("[INFO] TestTrainSplit - Writing transcripts at: %s | %s" % (f_trans_train_path, f_trans_test_path))
        f_trans_train = open(f_trans_train_path, "w")
        f_trans_test = open(f_trans_test_path, "w")
        f_trans_train.write(trans_train_content)
        f_trans_test.write(trans_test_content)
        f_trans_train.close()
        f_trans_test.close()

  print("[INFO] TestTrainSplit - Successfully processed %d files with %.2f train split." % (total_files, train_percentage))
  print("                        Train files: %d" % train_files)
  print("                        Test files: %d" % test_files)

if __name__ == "__main__":
  execute_split()