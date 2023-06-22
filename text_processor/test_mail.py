from cgitb import text
import spacy
from email.parser import BytesParser
from email import policy
import email
from html2text import HTML2Text
from gensim.parsing.preprocessing import STOPWORDS
import re
import logging
import langdetect
import os
import sys

# region Constants
# TODO Use the propoer filepath for the log
LOG_PATH = "example.log"
LOG_PREFIX = "MAIL_EXTRACTOR"
# endregion
# region variables
spacy_language_models = {
    "en": "en_core_web_sm",
    "fr": "",
    "de": "",
    "es": ""
}

def setup_logging(config: dict):
    """ This function will setup the loggging to have the desired logging when importing the package

    Args:
        config (dict): A dictionary with the filename, the logging level and the format of the entries
    """
    logging.basicConfig(
        filename=config["filename"],
        level=config["level"],
        format=config["format"]
    )    

blacklist = {}
# endregion
# region Functions
def detect_language(text: str) -> str:
    """ This function will return the most probably detected supported language or empty if it is not supported

    The function detects all possible languages that the input text is written on and returns the result with the highest probability
    as long as it is included in the list of supported languages 'spacy_language_models'. 

    Args:
        text (str): The input text that we want to guess the language it is written on.

    Returns:
        str: The language in ISO format that has the highest probability in detection or an empty string if that language is not supported.
    """
    results = langdetect.detect_langs(text)
    for result in results:
        if result.lang in spacy_language_models:
            return result.lang
    return ""

def clean_special_characters(text: str, replace_url: bool=True, replace_mail: bool=True, replace_filenames: bool=True) -> str:
    """
    Cleans a text from any special characters and returns only the words. This characters are punctuation
    marks or symbols, not letters from different alphabets. There is the option to also tokenize URLs, filenames, and mail addresses.

    Args:
    1. text (str): The text that will be cleansed
    2. replace_url (bool): Whether the function should also replace URLs for a unique token to minimize variation in the corpus
    3. replace_mail (bool): Whether the function should also replace mail addresses for a unique token to minimize variation in the corpus
    4. replace_filenames (bool): Whether the function should also replace filenames for a unique token to minimize variation in the corpus

    Returns:
    str: The new text without special characters and with the replaced token words if configured for it.
    """
    url_pattern = "(http.?:\/\/[^\s|^>]+)"
    mail_pattern = "[\w.+-]+@[\w-]+\.[\w.-]+"
    file_pattern = "[\w.+-]+\.[\w.-]+"
    new_text = re.sub("\\\\n|\\n|\\\\t|\\t", " ", text)
    new_text = re.sub("\\\\r|\\r", "", new_text)
    new_text = re.sub("--+|__+", "", new_text)
    if replace_mail:
        new_text = re.sub(mail_pattern, " hse_replaced_mail ", new_text)
    if replace_url:
        new_text = re.sub(url_pattern, " hse_replaced_url ", new_text)
    if replace_filenames:
        new_text = re.sub(file_pattern, " hse_replaced_filename ", new_text)
    words = new_text.split()
    return " ".join([re.sub('\W+', '', word) for word in words])

def get_text_from_msg_parts(parts: list) -> str:
    """ This function will retrieve the text inside a list of email.MEssage parts

    From the list it will iterate over them until it finds a part that is of main type 'text'. 
    It can be 'html' or 'plain' but it will still get the text within said part and return it as the answer.

    Args:
        parts (list): the list of email.Message parts that will be iterated over

    Returns:
        str: The first text found in the list of parts. (Only the first as continuing to iterate has duplicated the text within the email)
    """
    h = HTML2Text()
    h.ignore_images = True
    h.ignore_links = True
    h.ignore_tables = True
    h.ignore_emphasis = True
    text = ""
    for part in parts:
        if part.is_multipart():
            temp_parts = [p for p in part.walk()]
            text += "\n " + get_text_from_msg_parts(temp_parts)
        elif part.get_content_maintype() == "text":
            text += h.handle(str(part.get_payload(decode=True)))
            # Startgin from the third characters as all payloads convert the string to the format 'b "CONTENT"
            return text.lower()[2:]
    # Startgin from the third characters as all payloads convert the string to the format 'b "CONTENT"
    return text[2:]

def get_text_parts_of_eml_file(file_path: str, data: bytes) -> list:
    """ This function reads an EML file and gets the list of email.Message entities out of it

    Args:
        file_path (str): The path to the EML file in string format

    Returns:
        list: the resulting list of email.Message of type text retrieved from the parsed file.
    """
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            try:
                msg = BytesParser(policy=policy.default).parse(f)
            except Exception as e:
                logging.info(f"{LOG_PREFIX}: Failed to parse the EML file with the BytesParser. {e}")
                msg = email.message_from_binary_file(f)
    else:
        # the filepath does not exist, parsing the message parts from the binary data
        try:
            msg = BytesParser(policy=policy.default).parse(data)
        except Exception as e:
            logging.info(f"{LOG_PREFIX}: Failed to parse the EML file with the BytesParser. {e}")
            msg = email.message_from_bytes(data.getvalue())
    return [part for part in msg.walk() if part.get_content_maintype() == "text"]
# endregion
# region Classes
class TextProcessor:
    def __init__(self):
        """ At creation the TextProcessor class will start every spacy nlp tool from the dictionary of spacy_language_models"""
        self.spacy_tools = {key: spacy.load(spacy_language_models[value]) for key, value in spacy_language_models.items()}

    def text_processing(self, text: str, blacklist: dict={}, language: str="en", replace_names: bool=True, replace_tel: bool=True) -> str:
        """ This function processes the text and removes entities if specified.

        This function receives a text and some optional parameters that will affect the processing task. The processing will take a text and 
        might replace some words for general tokens. These can be names, like persons or company names, and/or numbers, these can be telephone numbers
        or numerical IDs. Additionally, the function might remove words entirely from the text if they are present in the blacklist parameter. The result 
        would be the text with the remaining lemmatized words and replacement tokens if specified.

        Args:
            text (str): The text that is going to be processed
            blacklist (dict, optional): This is a dictionary that contains as keys words that are forbidden to be in the result text. Defaults to {}.
            language (str, optional): The language that should be used to process the text in ISO format. Defaults to "en".
            replace_names (bool, optional): Whether the processor should replace names for a token to reduce variance in the corpus. Defaults to True.
            replace_tel (bool, optional): Whether the processor should replace or not the numbers in the text to reduce variance in corpus. Defaults to True.

        Returns:
            str: The resulting processed text without blacklisted words and with lemmatized words in it.
        """
        spacy_output = self.spacy_tools[language](text)
        words = []
        for token in spacy_output:
            if str(token) in blacklist:
                continue
            elif replace_names and token.ent_type_:
                words.append(" hse_replaced_name ")
            elif replace_tel and token.like_num:
                words.append(" hse_replaced_num ")
            else:
                words.append(token.lemma_)
        return " ".join(words)

    def get_text_from_eml(self, filepath:str=None, data: bytes=None, blacklist: dict={}, replace_names: bool=True, replace_tel: bool=True) -> str:
        """ This function will return the processed text from an EML file

        The function will retrieve the text within an EML file. This file can be input directly as bytes or with its filepath. From the 
        retrieved text, the processor will identify the language. If it is written in a supported language then it will proceed to use
        the spacy nlp tool and get the processed text. This result text is after removing stopwords, words included in the blacklist dictionary and 
        lemmatized.

        Args:
            filepath (str, optional): The full filepath to the EML input. Defaults to None.
            data (bytes, optional): The raw bytes read from the EML input file. Defaults to None.
            blacklist (dict, optional): A dictionray of words that should be removed from the retrieved text. Defaults to {}.
            replace_names (bool, optional): A boolean flag that indicates wether entity names within the text should be replaced or not. Defaults to True.
            replace_tel (bool, optional): A boolean flag that indicates wether numbers within the text should be replaced or not. Defaults to True.

        Returns:
            str: The result text in string format or empty if there was an error retrieving text from the EML file.
        """
        if filepath is None and data is None:
            return ""
        text = get_text_from_msg_parts(get_text_parts_of_eml_file(filepath, data))
        if text.strip() == "":
            logging.info(f"{LOG_PREFIX}: Failed to retrieve any text form the mail")
            return ""
        language = detect_language(text)
        if language == "":
            logging.info(f"{LOG_PREFIX}: The text in this mail is in an unsupported language")
            return ""
        return self.text_processing(text, blacklist, language, replace_names, replace_tel)

# endregion

if __name__ == "__main__":    
    pass
