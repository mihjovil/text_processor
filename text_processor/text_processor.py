from warnings import WarningMessage
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import langdetect
from gensim.corpora import Dictionary
import typing
from dataclasses import dataclass
import spacy
import pickle

 # region Constants
language_spacy_files = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm"
}
# endregion

@dataclass(order=False)
class BagOfWords:
    bow: typing.List[typing.Tuple[int, int]]
    dictionary: Dictionary

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, path: str):
        with open(path, 'rb') as file:
            self = pickle.load(file)
        


class TextProcessor:

    def initialize_spacy_tool(self) -> None:
        """This function takes the language from the TextProcessor and creates a Spacy NLP tool which can be used for processing text
        Raises:
            WarningMessage: If the language in the TextProcessor does not appear in the dictionary of languages in this file
        """
        if self.language not in language_spacy_files.keys():
            raise WarningMessage(f'The specified language {self.language} is not currently supported in our Spacy processing')
        else:
            spacy.cli.download(language_spacy_files[self.language])
            self.spacy_tool = spacy.load(language_spacy_files[self.language])

    def __init__(self, language: str=None) -> None:
        """The initializer can receive the language in string format (abbreviated) in order to initialize the NLP tool of the TextProcessor

        Args:
            language (str, optional): A string indicating which language the TextProcessor will initialize its NLP tool. This has to be abbreviated in ISO 639-1 form. Defaults to None.
        """
        if language is not None:
            self.language = language            
            self.initialize_spacy_tool()
    
    def simple_preprocess(self, text:str, minimum_length: int=3, blacklist: typing.Dict = {}, ignore_list: typing.Dict = {}) -> typing.List[str]:
        """This function will perform a simple processing of text by removing accents and symbols from the text

        Args:
            text (str): The text that will be processed to remove the simple stopwrods and accents from a word
            minimum_length (int, optional): the minimum length that words in the text must have to remain in it. Defaults to 3.
            blacklist (typing.Dict): A dictionary of words that should not be allowed after processing. 

        Returns:
            typing.List[str]: A list with every word that remains from the input text and fulfills all the filters
        """
        text = gensim.utils.simple_preprocess(text)
        answer = []
        for word in text:
            if word in blacklist.keys():
                break
            elif word not in STOPWORDS and len(word)>minimum_length and word not in ignore_list.keys() and not word.__contains__('_'):
                answer.append(word)
        return answer

    def detect_language(self, text: str) -> typing.Tuple[str, float]:
        """This function takes a text and tries to predict the language it is written in.
        It returns the most likely language it is written in an a probability of being this language (0 to 1).
        Additionally if the TextProcessor has not been fully initialized (no language was specified in the constructor) it uses the detected language to initialize its own NLP tool.

        Args:
            text (str): The text that is used as input to detect the language it is written in.

        Returns:
            typing.Tuple[str, float]: A tuple that contains the detected language in string format (in abbreviated form 'en' for english) and the probability of the text 
            being written in that language.
        """
        detections = langdetect.detect_langs(text)
        most_likely = detections[0]
        language, probability = most_likely.lang, most_likely.prob
        # Check if the processor has been fully initialized or not
        if self.language is None:
            self.language = language            
            self.initialize_spacy_tool()
        return language, probability

    def spacy_preprocess(self, text: str, minimum_length:int=3, blacklist: typing.Dict = {}, ignore_list: dict = {}) -> typing.List[str]:
        """This function takes a text as input and processes it. This means that it lemmatizes all words inside of it using the Spacy NLP tool, removes the stopwords and also the words with less than
        the specified minimum length. If no minimum length is specified, it defaults to three letters.

        Args:
            text (str): The text that will be processed. It must be a string value.
            minimum_length (int, optional): The minimum amount of characters a word must have not to be filtered by the processor. Defaults to 3.
            blacklist (typing.Dict): A dictionary of words that should not be allowed after processing. 

        Returns:
            typing.List[str]: Returns a list of string with all the words that remain in the text after processing.
        """
        nlp_results = self.spacy_tool(text)
        words = " ".join([w.lemma_ for w in nlp_results])
        return self.simple_preprocess(words, minimum_length, blacklist=blacklist, ignore_list=ignore_list)

    def create_bag_of_words(self, documents: typing.List[typing.List[str]], filter_below: int=15, filter_above: float=0.1, keep: int=100000) -> BagOfWords:
        """ This function will use a list of documents (should have been preprocessed) to create a Bag Of Words and a dictionary that has the value of the word mapped to the unique int id.
        This two resulting structures come very handy when using LDA.

        Args:
            documents (typing.List[typing.List[str]]): A list of preprocessed documents that will be used to create the BOW. It must be a two dimensional iterable.
            filter_above (float, optional): The proportion (percentage) of words that will be filtered if they are present in any percentage higher of documents higher than this. Defaults to 0.5.
            filter_below (int, optional): The minimum number of docuemnts a word must be in in order to be kept. Defaults to 5.
            keep (int, optional): The maximum number of words that will be kept for the corpus. Defaults to 1e4.

        Returns:
            BagOfWords: A dataclass that contains the actual bag of words made form the documents list as well as a dictionary to map the id of a word to its string value
        """
        # Create a Dictionary
        dictionary = Dictionary(documents=documents)
        # Filter words inside of dictionary        
        dictionary.filter_extremes(no_below=filter_below, no_above=filter_above, keep_n=keep)
        bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
        self.bow = BagOfWords(bow=bow_corpus, dictionary=dictionary)
        return self.bow

if __name__ == '__main__':
    print('Main of TextProcessor')
