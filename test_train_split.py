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

from params_data import *

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
      full_dir = root + "/" + dir
      dir_files = os.listdir(full_dir)
      dir_files = [full_dir + "/" + file for file in dir_files] 

      wav_files = []
      for dir_file in dir_files:
        if dir_file.endswith(".wav"):
          dir_file = dir_file.replace("\\", "/")
          wav_files.append(dir_file)
          total_files += 1
      
      # Split the wav files into train/test and copy to respective 
      # locations. 
      if len(wav_files) > 0:
        print("[INFO] TestTrainSplit - Processing %s" % full_dir)
        train, test = np.split(wav_files, [int(len(wav_files)*train_percentage)])
        old_train = train.copy()
        old_test = test.copy()
        new_train = []
        new_test = []
        for file in train: new_train.append(file.replace(output_folder, train_location))
        for file in test: new_test.append(file.replace(output_folder, test_location))
        train_path = []
        test_path = []
        for file in new_train: train_path.append(file.rsplit('/', 1)[0])
        for file in new_test: test_path.append(file.rsplit('/', 1)[0])

        # We now have full paths for all of our new files. Copy.
        for i in range(0, len(new_train)):
          os.makedirs(train_path[i], exist_ok=True)
          shutil.copyfile(old_train[i], new_train[i])
          train_files += 1
        
        for i in range(0, len(new_test)):
          os.makedirs(test_path[i], exist_ok=True)
          shutil.copyfile(old_test[i], new_test[i])
          test_files += 1

  print("[INFO] TestTrainSplit - Successfully processed %d files with %.2f train split." % (total_files, train_percentage))
  print("                        Train files: %d" % train_files)
  print("                        Test files: %d" % test_files)

if __name__ == "__main__":
  execute_split()