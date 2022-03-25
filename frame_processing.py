#
# frame_processing.py
#
# Given a singular frame, work through the associated segment of
# audio activity to determine if it is the beginning of a new
# utterance and the end of an old one or is part of the previous
# frame's utterance. 

from frame_utils import *
from params_data import *
from audio_utils import *
from speaker_whitelist import speaker_whitelist

import pytesseract
from difflib import SequenceMatcher
import os

# Enums to make behavior clearer.
NO_AUDIO_ACTIVITY = 1 # VAD says this frame has no activity. Move on. ]
NO_SPEAKER_FOUND = 2
NO_TEXT_FOUND = 3
SAME_UTTERANCE = 4 # VAD says this frame is the same as the current utterance. Move on.
DROP_UTTERANCE = 5
NEW_UTTERANCE_BAD = 6 # A valid utterance, but not accepted transcript. (unknown speaker, bad text)
NEW_UTTERANCE_GOOD = 7 # A new utterance. 

def process_frame(wav, video_id, frame, frame_num, video_length, 
                   activity_segments, activity_index, prev_transcript, 
                   prev_speaker, prev_drop_utterance, prev_start, game_name,
                   statistics, speaker_blacklist, speaker_indices, cleaner,
                   transcripts, visualization=False, multiprocessing = False):
  """
  Provided information about the current frame, the current frame
  itself, as well as information regarding the "current" utterance,
  process the frame with OCR. Manage and preprocess the result if it
  is a new utterance. 

  Can return in a few ways: 
  - (status, prev_do_not_save, prev_start) -> not a new utterance.
  - (status, prev_do_not_save, prev_start, prev_transcript, prev_speaker) -> new utterance (good/bad)
  """
  # First, preprocess the frames. Get Regions of Interest and then 
  # process each region for the stuff we need for OCR.
  subtitle_roi = get_frame_region_of_interest(frame, game_name)
  speaker_roi = get_frame_speaker_region(frame, game_name)
  subtitle_roi_preprocessed = preprocess_frame(subtitle_roi, game_name, True)
  speaker_roi_preprocessed = preprocess_frame(speaker_roi, game_name, False)

  # Show the entire preprocessed frame if visualizing.
  if visualization: frame = preprocess_frame(frame, game_name)

  new_utterance = None
  drop_current_utterance = False

  # OCR. First, read the speaker name. Preprocess the name so
  # that we can look it up in the game's speaker name whitelist. 
  speaker_name = pytesseract.image_to_string(speaker_roi_preprocessed)
  speaker_name, drop_current_utterance,cleaner_dropped = cleaner.preprocess_text(speaker_name, drop_current_utterance)
  if cleaner_dropped: statistics["total_cleaner"] += 1

  # If no speaker is found, this is considered a NEW utterance and
  # the previous utterance (if present) to be complete. 
  if speaker_name is None or speaker_name == "":
    _print_text("[WARNING] Dataset - V-%d - No speaker found!" % video_id, multiprocessing)
    speaker_name = ""
    if visualization: debug_frame_view(speaker_roi_preprocessed, "New (%s): \"%s\"" % (speaker_name, ""), "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript), "WARNING - NO SPEAKER FOUND!")
    new_utterance = True
    drop_current_utterance = True
    statistics["total_no_speaker"] += 1

  # If the speaker name is different, we don't even need to compare 
  # the transcript - we know it's a new utterance. OCR the trancript.
  # Otherwise if the same speaker, OCR the transcript and compare to
  # previous transcript. If it's the same, same.
  if prev_speaker is None or prev_speaker != speaker_name:
    new_utterance = True

  subtitles = pytesseract.image_to_string(subtitle_roi_preprocessed)
  subtitles, drop_current_utterance, cleaner_dropped = cleaner.preprocess_text(subtitles, drop_current_utterance)
  if cleaner_dropped: statistics["total_cleaner"] += 1

  # Toss out any frames without subtitles. 
  if subtitles is None or subtitles == "":
    # Warn the user. But if we already know this utterance is bad, do
    # not bother the user - we ale already treating this utterance as
    # a new, bad utterance.
    subtitles = ""
    if drop_current_utterance is False:
      _print_text("[WARNING] Dataset - V-%d - No transcript found!" % video_id, multiprocessing)
      if visualization: debug_frame_view(subtitle_roi_preprocessed, "New (%s): \"%s\"" % (speaker_name, subtitles), "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript), "WARNING - NO TRANSCRIPT FOUND!")
      statistics["total_no_transcript"] += 1
    new_utterance = True
    drop_current_utterance = True

  if new_utterance is None:
    # Time to compare transcripts. We will specifically parse the % that
    # the current transcript matches the previous. under a certain 
    # threshold, the previous transcript wil be assumed to be equal 
    # to the existing one, as the chances of the same speaker uttering
    # the nearly same (but not different) text directly after uttering
    # it is assumed to be near zero. 
    variance_from_prev_transcript = SequenceMatcher(None, subtitles, prev_transcript).ratio()

    if variance_from_prev_transcript <= subtitle_variance_thresh:
      new_utterance = True
    else:
      new_utterance = False
      # Don't bother check if we already know this utterrance is corrupted. 
      if variance_from_prev_transcript <= 1.0 - subtitle_variance_acceptable_thresh:
        if prev_drop_utterance is False:
          _print_text("[WARNING] Dataset - V-%d - Potential discrepancy found! Prev start: %d" % (video_id, prev_start), multiprocessing)
          print("                    Prev (%s): \"%s\"" % (prev_speaker, prev_transcript.replace("\n", " ")))
          print("                    Current (%s): \"%s\"" % (speaker_name, subtitles.replace("\n", " ")))
          print("                    Similarity: %.2f" % variance_from_prev_transcript)
          #if visualization: _debug_frame_view(subtitle_roi_preprocessed, "New (%s): \"%s\"" % (speaker_name, subtitles), "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript), "WARNING - %.2f MATCH!" % variance_from_prev_transcript)
          statistics["total_discrepancies"] += 1

        # We don't consider this to be a new utterance if this is the case. 
        # Mark the current utterance as corrupted and continue looking
        # for it, so we know when it ends. 
        drop_current_utterance = True

  # Make sure the speaker name is whitelisted for this particular 
  # game name. If not, alert user (if first occurence). This is 
  # a bad utterance. 
  if speaker_name != "" and speaker_name not in speaker_whitelist[game_name]:
    if speaker_name not in speaker_blacklist:
      _print_text("[INFO] Dataset - V-%d -Encountered non-whitelisted character %s." % (video_id, speaker_name), multiprocessing)
      speaker_blacklist.append(speaker_name)
      if visualization: debug_frame_view(speaker_roi_preprocessed, "New (%s): \"%s\"" % (speaker_name, ""), "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript), "UNKNOWN SPEAKER FOUND.")
    statistics["total_blacklisted_speaker"] += 1
    drop_current_utterance = True

  if new_utterance is False:
    # Manage the possibility of a corrupted utterance. In this case,
    # (discrepancy with text), turn on drop flag. 
    if drop_current_utterance is True:
      return(transcripts, statistics,speaker_blacklist, speaker_indices, DROP_UTTERANCE, True, prev_start, subtitles, speaker_name)
    else:
      return(transcripts, statistics, speaker_blacklist, speaker_indices, SAME_UTTERANCE, prev_drop_utterance, prev_start)

  # At this point we KNOW this is now a NEW UTTERANCE. Wrap up the 
  # previous utterance. Write the wav to file and append the 
  # transcript to the speaker's transcript file for this video_id. 
  complete_utterance_start = prev_start
  complete_utterance_end = activity_segments[activity_index-1][1]
  complete_utterance_speaker = prev_speaker
  if not prev_drop_utterance:
    # Skip the first period of silence.
    if complete_utterance_speaker is not None:
      # Process the wav first.
      if complete_utterance_speaker not in speaker_indices:
        speaker_indices[complete_utterance_speaker] = 0
      else:
        speaker_indices[complete_utterance_speaker] += 1

      # EX: TalesSkits/full / VELVET / 1 / VELVET-1-0001
      new_wav_fpath = output_folder + "/" + complete_utterance_speaker + "/" + str(video_id)
      os.makedirs(new_wav_fpath, exist_ok=True)
      wav_filename = complete_utterance_speaker + "-" + str(video_id) + "-" + f'{speaker_indices[complete_utterance_speaker]:04}.' + output_format
      full_wav_path = new_wav_fpath + "/" + wav_filename

      # Process. 
      extract_new_wav(original = wav,
                      start = complete_utterance_start,
                      end = complete_utterance_end,
                      fpath = full_wav_path)

      # Now manage the transcript. Tje transcript information is stored
      # in a dict like so:
      # transcripts = {
      #   "ROLLO" : ((filename, transcript), folder_path)
      # }
      complete_utterance_transcript = prev_transcript
      if complete_utterance_speaker not in transcripts:
        transcripts[complete_utterance_speaker] = ([], new_wav_fpath)
      
      transcript_tuple = (wav_filename, complete_utterance_transcript)
      transcripts[complete_utterance_speaker][0].append(transcript_tuple)

      statistics["successful_utterances"] += 1
  else:
    # Don't flood the console with reports of dropped blacklist characters. 
    # No action needs to be taken for these. 
    statistics["total_dropped"] += 1
    if not (complete_utterance_speaker != "" and complete_utterance_speaker not in speaker_whitelist[game_name]):
      _print_text("[DEBUG] Dataset - V-%d -Dropped utterance (%s) with range: %d - %d" 
        % (video_id, complete_utterance_speaker, complete_utterance_start, complete_utterance_end), multiprocessing)


  # If the speaker name is whitelisted, process the transcript with
  # preprocessing to match LibriSpeech format. If the transcript 
  # ends up empty, this is a bad utterance.

  # If everything is good up until this point, we're all set. 
    
  start = activity_segments[activity_index][0]
  if visualization: 
    end = activity_segments[activity_index][1]
    middle = int(((end - start)//2 ) + start)

    text_line_1, text_line_2, text_line_3 = None, None, None
    text_line_1 = "New (%s): \"%s\"" % (speaker_name, subtitles)
    if prev_start is not None and prev_speaker is not None:
      text_line_2 = "Prev (%s): \"%s\"" % (prev_speaker, prev_transcript)
      info_tuple = (str(prev_drop_utterance), prev_start, activity_segments[activity_index-1][1], activity_index,start, end, middle, frame_num)
      text_line_3 = "Prev Drop: %s | Prev Start: %d | Prev End: %d | AI: %d | Start: %d | End: %d | Middle: %d | Frame: %d" % info_tuple
    debug_frame_view(speaker_roi_preprocessed, text_line_1=text_line_1, text_line_2 = text_line_2, text_line_3 = text_line_3)

  return(transcripts, statistics, speaker_blacklist, speaker_indices, NEW_UTTERANCE_GOOD, drop_current_utterance, start, subtitles, speaker_name)

def _print_text(output, multiprocessing):
  """
  If we are printing out the progress bar for each step, use a
  newline before each message.
  """
  if multiprocessing:
    print(output)
  else:
    print("\n" + output)