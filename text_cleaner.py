#
# text_cleaner
#
# Preprocesses text given original from the OCR parsed frame during
# dataset generation. Relatively harmonized with cleaning done with
# english cleaners in multispeaker synthesis. 

import re
from unidecode import unidecode

# All whitespace
_whitespace_re = re.compile(r"\s+")

_stutter_re = re.compile(r"[A-Za-z]-[A-Za-z]")

# All symbols that are sentence breaks that are NOT periods.
_sentence_break_re = re.compile(r"[,|!|?]")

# All symbols that are NOT white space, or acceptable sentence
# splits. 
_symbols_re = re.compile(r"[^\w|\s|'|.|,|!|?]")

class Cleaner: 
  _last_message = ""

  def __init__(self, game_name):
    self.game_name = game_name

  def preprocess_text(self, original_text, drop_current_utterance):
    """
    Given original text from the OCR parsed frame, use a harmonized
    text preprocessing method for both speakers and transcripts. 
    Invalidate provided text here. Returns both text as well as
    the invalidate text flag.

    - Utterances including * are invalid.
    - Utterances including "[A-Za-z]-[A-Zaz]" are invalid ("R-right")
    - Replace | with I
    - Replace all newlines with space.
    - Replace all utterance breaks with . (! ? , . etc)
    - Manage potential Unicode characters: ‘’ “”  ‟ 🙶 🙷
    - Remove all silence periods (...) 
    - Remove leading/trailing whitespace
    - All characters to upper.
    """

    cleaner_dropped = False

    if drop_current_utterance is False and "*" in original_text:
      message_to_print = "\n[WARNING] Text Cleaner - Submitted text contains \"*\" symbol. Marking as invalid.\n"
      message_to_print += "                         Text: \"%s\"" % original_text.replace("\n", "")
      if message_to_print != self._last_message:
        print(message_to_print)
        self._last_message = message_to_print
      drop_current_utterance = True
      cleaner_dropped = True
    
    if drop_current_utterance is False and re.search(_stutter_re,original_text) is not None:
      message_to_print = "\n[WARNING] Text Cleaner - Submitted text contains \"[A-Z]-[A-Z]\" (stutter). Marking as invalid.\n"
      message_to_print += "                         Text: \"%s\"" % original_text.replace("\n", "")
      if message_to_print != self._last_message:
        print(message_to_print)
        self._last_message = message_to_print
      drop_current_utterance = True
      cleaner_dropped = True
    
    processed_text = original_text
    processed_text = processed_text.replace("|", "I")
    processed_text = processed_text.replace("\n", " ")
    processed_text = self._convert_to_ascii(processed_text)
    processed_text = re.sub(_sentence_break_re, ".", processed_text)
    processed_text = re.sub(_symbols_re, "", processed_text)
    processed_text = processed_text.replace("_","")
    processed_text = self._replace_up_to_x_repeated(".", 10, processed_text)
    processed_text = self._collapse_whitespace(processed_text)
    processed_text = self._uppercase(processed_text)
    processed_text = processed_text.strip()

    # Per game behavior. 
    if self.game_name == "berseria":
      processed_text = processed_text.replace("1", "I")

    return str(processed_text), drop_current_utterance, cleaner_dropped

  def _replace_up_to_x_repeated(self, repeated_char, x, text):
    """
    For a given char (Ex) . ), replace sequences of that char repeated
    for up to x times. Kinda brute force.
    """
    for i in range(x, 1, -1):
      test_string = ""
      for j in range(0, i): test_string += repeated_char
      if test_string in text:
        text = text.replace(test_string, repeated_char)
    
    return text


  def _uppercase(self, text):
    return text.upper()

  def _collapse_whitespace(self, text):
    """Normalize whitespace, so >=double-spaces turn into singular spaces."""
    return re.sub(_whitespace_re, " ", text)

  def _convert_to_ascii(self, text):
    return unidecode(text)


# For debug purposes only. 
if __name__ == "__main__":
  solution = preprocess_text("‘’ “”                         ‟ 🙶 🙷.......?!!@?JFIOJ#asefasef!)(@&#)(%)#*(()!#()$%*)(!()@#%$(~!@!@$#!@#??/!!@)#%_*(@^@&#$*($)_@!$)@!&%$*()!@*$@!$!@$@!$@(!.HELLO!@#$%&^!,...,..../??!@#@!@?#$@!J#@!#@!?!!!...", False)
  # Expect: Solution: ("'' .JFIOJASEFASEF.HELLO.J.", True)
  print("\nSolution: ", end="")
  print(solution)