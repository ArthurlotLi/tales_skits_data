
# Tales Skits Data - Dataset Collection Algorithms

Tools for collecting voice actor data from Tales of video games. Produces audio samples and transcripts of utterances grouped by speaker, identified and transcribed via Optical Character Recognition of on-screen texts during ingame "skits".

Data is preprocessed in a manner that makes it a suitable companion for the LibriSpeech dataset.

[![Tales of Skits Website](https://i.imgur.com/A7HdMCQ.png "Tales of Skits Website")](http://talesofskits.com/)

Integrates the speaker encoder model from the multispeaker_synthesis project in order to detect obviously "unclean" utterances by comparing all samples in a directory to each other and flagging utterances with a diff vector that has a greater l2 norm than the specified tolerance. 

This is necessary as there are inherently undetectable discrepancies when correlating speakers to utterances - such as speaker utterances purposefully stepping on each other, multiple speakers speaking at the same time under a single name, small uncharacteristic utterances, or even speakers posing as other speakers. (Teepo ventriloquism, for example) 

Data generation variables such as sampling rate, error tolerances, etc. are highly configurable, found in data_params.py.

Once the data has been generated, for use in the multispeaker_synthesis project a few addtional steps are recommended. 

1. Split dataset into test and train (by utterance, not speaker)
2. Preprocess data and transcripts for Montreal Force Aligner
3. Use Montreal Force Aligner to generate TextGrids
4. Postprocess using TextGrids, wavs, and transcripts to generate alignment files. 

For more information, please see [Tales of Skits](http://talesofskits.com/).

[![Tales of Skits Website](https://i.imgur.com/9HlmT9X.png "Tales of Skits Website")](http://talesofskits.com/)

---

### Usage:

To generate a dataset, make sure to do the following:

1. Install FFmpeg. The pip installation won't set the cli tool, so you need to install from here:

   https://ffmpeg.org/

   Make sure to add the directory with ffmpeg.exe to the system PATH, then restart your command line/terminal session.

2. Install requirements by running:

   pip install -r requirements.txt

3. If you don't have it, install tesseract for your platform and place on the system PATH so that entering "tesseract" in the command line or terminal doesn't throw an error. 

   https://tesseract-ocr.github.io/tessdoc/Home.html

4. Ensure that multispeaker_synthesis is present in the parent directory one level up from this one. If you don't want to use speaker verification as post processing (it's highly recommended), disable the feature in data_params. 

   You can also change other speaker verification parameters there, like which model iteration to use (it's location), and what the acceptable threshold for l2 normed difference is. 

5. Gather .mp4 files and place in a subfolder called "data" in the repo.

6. Run dataset_generation.py. Output data will be placed in a folder titled "TalesSkits".

7. Optionally, run test_train_split.py to split data accordingly. Note that the split occurs by utterances, not by speakers, so the same speakers will appear in test and train.

8. To generate alignment files, follow these additional steps after 
   splitting into test and train.

   a. Copy test/train directories to TalesSkitsAligned

   b. Run test_train_align_preprocess.py

   c. If on windows, use the linux subsystem with wsl --install on powershell.

   d. Install conda, MFA, activate MFA environment in the subsystem

   e. Navigate to tales_skits_data via mnt folder

   f. Run mfa download for acoustic english and dictionary english

   g. Run $ mfa validate TalesSkitsAligned english english

   h. Verify everything is good, then run $ mfa align TalesSkitsAligned english english
   i. Access the files on windows via file explorer //wsl$. Copy the files into the TalesSkitsAligned directory and rename as train_textgrids and test_textgrids
   j. Run postprocessing - test_train_align_postprocess.py
   k. Copy the resulting files directly into your original dataset and voila, you have alignment files, ready for multispeaker synthesis synthesizer preprocessing! 