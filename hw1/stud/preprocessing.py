from typing import List
import re
import unicodedata


special_symbols = ["؛", "۔", "،", "。", "؟", "।", "、", 
                    "，", "¡", "¿", "…", "‡","×", "†", 
                    "√", "·", "→", "«", "»", "™", "′", 
                    "•", "„", "½", "¼", "¾", "∞", "θ",
                    "∅", "²", "³"]

dashes = ['–', "―", "—"]
translate_dashes = {dash: '-' for dash in dashes}
quotes = ["‘","’", "“", "”", "„", "\""]
translate_quotes = {quote: '\'' for quote in quotes}

currencies = ['£', '€', '¥', '฿', '₽', '﷼', '₴', '₠', '₡', '₢', '₣', '₤', '₥', '₦', '₧', '₨', '₩', '₪', '₫', '€', '₭', '₮', '₯', '₰', '₱', '₲', '₳', '₴', '₵', '₶', '₷', '₸', '₹', '₺', '₻', '₼', '₽', '₾', '₿']
translate_currencies = {c: '$' for c in currencies}



def convert_in_ascii(string: str) -> str:
    """
    Returns
    -------
        An ASCII string, where some non-ASCII characters
        are replaced with the corrisponding ascii (e.g., è --> e).
        If the string contains only non-ASCII characters, it
        is substituted with an undescore '_'.
        (e.g., chinese chars are replaced with '_').
        
        If the input is a single non-ASCII character, it
        is replaced with the question mark '?'.
        
    Parameters
    ----------
    string: str
        A string.
    """
    replace_special_symbols = {sym: "?" for sym in special_symbols}
    char_to_be_translated = {
                            **translate_dashes,
                            **translate_quotes,
                            **translate_currencies,
                            **replace_special_symbols
                            }
    
    # Single char
    if len(string) == 1:
        string = string.translate(str.maketrans(char_to_be_translated))     
        
    in_ascii = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode()
        
    # If the string is not convertible with any ASCII char
    if in_ascii == "":
        in_ascii = "_"
    
    return in_ascii

remove_special_symbols = {sym: '' for sym in special_symbols}

# Remove parenthesis when they are present with the token
# e.g., (house) --> house
remove_parenthesis = {
    '(': '',
    ')': '',
    '[': '',
    '{': '',
    ']': '',
    '}': ''
}
char_to_be_translated = {
                        **translate_dashes, 
                        **translate_quotes,
                        **translate_currencies,
                        **remove_parenthesis,
                        **remove_special_symbols,
                        }

# Transform parenthesis when they are the only token ( len(token)==1 ) 
# e.g., [ --> (
translate_parenthesis = {
    '[': '(',
    '{': '(',
    ']': ')',
    '}': ')'
}


punc_to_remove_from_end = ",;:'?!+-"

# The input list is not modified
def clean_text(text: List[List[str]]) -> List[List[str]]:
    """
    Returns
    -------
        A cleaned list of lists of strings where each number is
        replaced with '1' and where special_symbols and non-ASCII
        characters are processed in order to get unifom data.
        
        The returned list is a new list, so the input is not
        touched.
        
    Parameters
    ----------
    text: List[List[str]]
        A list of lists of strings. 
        In this case each nested list is a sentence.
    """

    # Substitute each number with 1
    text = [re.sub("\d+", "1", token) for token in text] 
    
    # Remove punctuation contained in a token 
    # (e.g., if the tokens in the list are ["what" "time" "it" "is?"],
    # then after the transformation --> ["what" "time" "it" "is"])                        
    text = [token.translate(str.maketrans(char_to_be_translated)) if len(token)>1 else token for token in text] 
   
    
    text = [token.translate(str.maketrans(translate_parenthesis)) if len(token)==1 else token for token in text] 

    text_without_last_punc = ["_" for _ in range(len(text))]
    for idx, token in enumerate(text):
        text_without_last_punc[idx] = token
        
        if len(token) > 1:
            last_char = text_without_last_punc[idx][-1]
            while last_char in punc_to_remove_from_end:
                text_without_last_punc[idx] = text_without_last_punc[idx][:-1] 
                last_char = text_without_last_punc[idx][-1]

    text = text_without_last_punc

    # Convert non-ASCII characters in their corrisponding ascii (e.g, è --> e)
    text = [convert_in_ascii(token) if not token.isascii() else token for token in text]
      
    return text