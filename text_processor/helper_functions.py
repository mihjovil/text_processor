import os
from typing import List, Tuple, Any
from collections import Counter
import typing
import langdetect
from bs4 import BeautifulSoup
import email
from dataclasses import dataclass
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.metrics import confusion_matrix
from utils.text_processor import BagOfWords, TextProcessor
import gensim
from tqdm import tqdm
import traceback
from datetime import datetime
import re
from matplotlib import pyplot as plt
import seaborn as sns
import random
from email import policy
from email.parser import BytesParser
from unidecode import unidecode
import html2text

# region File processing
@dataclass
class DetectedLanguage:
    language: str
    probability: float
    text:str

@dataclass
class ExceptionCase:
    exception: str
    text: str
    def __str__(self) -> str:
        return f'CAUSE: {self.exception}. TEXT: {self.text}'
    
    
def clean_special_characters(text: str) -> str:
    """
    Cleans a text from any special characters and returns only the words. This characters are puntuation
    marks or symbols, not letters from different alphabets.

    Args:
    1. text (str): The text that will be cleansed

    Returns:
    str: The new text without special characters
    """
    words = text.split()
    url_pattern = "((http|https):\/\/([^\s]+))"
    words = [re.sub(url_pattern, " <<URL>> ", word) for word in words]
    return " ".join([re.sub('\W+', '', word) for word in words])

def skip_part(part):
    """
    Checks if given mailpart is attachment
    """

    if re.match(r'(\s)*?attachment.*', str(part['Content-Disposition'])):
        return True
    if part.get_content_type() != 'text/html' and part.get_content_type() != 'text/plain':
        return True
    if part.get_param('name'):
        return True
    return False

def get_text_from_html(html_text:str) -> str:    
    soup = BeautifulSoup(html_text, features = 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    # Removing links from text
    for a in soup.findAll('a', href=True):
        a.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def get_charset_from_file(path: str) -> str:
    file = open(path, encoding='ISO-8859-1')
    raw_text = file.read()
    file.close()
    message = email.message_from_string(raw_text)
    charsets = message.get_charsets()
    for c in charsets:
        if c is not None:
            return c
    return "ISO-8859-1"

def should_fetch_html(message: Any) -> bool:
    # message is type email.Message
    plain, html = 0, 0
    for part in message.walk():
        ctype = part.get_content_maintype()                                 
        if ctype == 'text':
            csubtype = part.get_content_subtype()
            if csubtype == 'plain':
                plain += 1
            elif csubtype == 'html':
                html += 1
    if plain == html or plain == 0 and html > 0:
        # It should take the HTML instead of the plain text
        return True
    return False

# region deprecated mail processing
def deprecated_get_relevant_info(path: str) -> typing.Tuple[str, bool]:
    file = open(path, encoding=get_charset_from_file(path))
    raw_text = file.read()
    file.close()
    message = email.message_from_string(raw_text)
    found_content = True
    body = ""
    if message.is_multipart():
        should_html = should_fetch_html(message)
        for part in message.walk():
            ctype = part.get_content_maintype()
            if ctype == 'text':
                csubtype = part.get_content_subtype()
                if csubtype == 'plain' and not should_html:
                    body += str(part.get_payload(decode=True), encoding='utf-8') + "\n\n"  # decode
                elif csubtype == 'html' and should_html:
                    body += get_text_from_html(str(part.get_payload(decode=True))) + "\n\n" 
                else: 
                    # Calendar is not addressed yet
                    continue
                break
    else:
        ctype = message.get_content_maintype()        
        if ctype == 'text':
            csubtype = message.get_content_subtype()
            if csubtype == 'plain':
                body = message.get_payload(decode=True)
            elif csubtype == 'html':
                body = get_text_from_html(message.get_payload())
    if body == "":
        # if there was no text found in the mail, we save the whole mail for further analysis
        body = raw_text  
        found_content = False
    body = re.sub(r'(\\n)', "", str(body))
    return str(body), found_content
# endregion

def skip_part(part):
    """
    Checks if given mailpart is attachment
    """

    if re.match(r'(\s)*?attachment.*', str(part['Content-Disposition'])):
        return True
    if part.get_content_type() != 'text/html' and part.get_content_type() != 'text/plain':
        return True
    if part.get_param('name'):
        return True
    return False

def get_relevant_info_mail(path: str) -> Tuple[str, bool]:    
    """
    This new function uses a different package to get the text from the mail. It returns always True in whether it found
    or not any content as it should always get content and for backwards compatibiliy purposes too.

    Args:
    1. path (str): The file path of the EML file that the function will get the text from

    Returns:
    Tuple[str, bool]: A tuple with the text from the mail (subject and body) plus a boolean informing that it found
    content inside the mail (deprecated value)
    """
    with open(path, 'rb') as fp:
        msg = BytesParser(policy=policy.default).parse(fp)
    try:
        h = html2text.HTML2Text()
        h.ignore_links = True
        body = msg.get_body()
        if msg["Subject"] is not None and body is not None and str(body.get_content()).strip() != "":
            answer = msg["Subject"] + "\n\n" + h.handle(body.get_content())
            faulty_pattern = r"(content-type|content-disposition|content-transfer-encoding)"        
            if len(re.findall(faulty_pattern, answer, re.IGNORECASE)) > 0:
                # It has the keywords inside the text. It is a faulty extraction
                return None, False
            return unidecode(answer), True
        else:
            return None, False
    except KeyError:
        try:
            with open(path, "rb") as f:
                msg = email.message_from_binary_file(f)
                for part in msg.walk():
                    if skip_part(part):
                        continue
                    if part.is_multipart():
                        continue
                    if part.get_charset() is None or part.get_charset() == "":
                        part_charset = 'utf-8'
                    else:
                        part_charset = part.get_charset()
                    part_payload = part.get_payload(decode=True)
                    content_type = part.get_content_type()
                    # Skipping when not HTML or plain text
                    if content_type not in ["text/html", "text/plain"]:
                        continue
                    try:
                        part_str = part_payload.decode(part_charset, errors='ignore')
                        if content_type == "text/html":
                            h = html2text.HTML2Text()
                            h.ignore_links = True
                            part_str = h.handle(part_str)
                        faulty_pattern = r"(content-type|content-disposition|content-transfer-encoding)"                        
                        if len(re.findall(faulty_pattern, part_str, re.IGNORECASE)) > 0:
                            # It has the keywords inside the text. It is a faulty extraction
                            return None, False
                        return unidecode(part_str), True                    
                    except Exception as ex:
                        continue                                                        
        except:
            return None, False
        return None, False
    except:
        return None, False

def get_file_paths(root: str) -> typing.List[str]:
    answer = []
    for path in os.listdir(root):
        if os.path.isdir(f'{root}/{path}'):
            answer.extend(get_file_paths(f'{root}/{path}'))
        else:
            answer.append(f'{root}/{path}')
    return answer

def detect_language(text: str) -> DetectedLanguage:
    languages = langdetect.detect_langs(text)
    answer_lang, answer_prob = None, 0
    for language in languages:
        if language.prob > answer_prob:
            answer_lang, answer_prob = language.lang, language.prob
    return DetectedLanguage(answer_lang, answer_prob, text)

def write_to_folder(folder:str, file_path: str, language: str, prob:float, text: str) -> None:
    filename = file_path.split('/')[-3:]
    filename = "_".join(filename)
    target_path = f'prob_{prob:.3f}_{filename}'
    target_directory = f'{folder}/{language}'
    if not os.path.isdir(target_directory):        
        # The directory needs to be created
        os.mkdir(target_directory)
    file = open(f'{target_directory}/{target_path}.txt', 'x')
    file.write(text)
    file.close()

def mail_processing(paths: List[str], languages_directory: str) -> Tuple[dict, int, str, str]:
    problematic_paths = ""
    exceptions = ""
    number_of_exceptions = 0
    languages = {}
    for p in tqdm(paths):
        try:
            temp_body, found_content = get_relevant_info_mail(p)            
            if not found_content:
                # These paths have been analyzed and they only posses files, no text that can be useful
                problematic_paths += f'Could not read body: {p}\n'
                continue
            elif len(temp_body) > 10_000:
                problematic_paths += f'The body is too long: {p}\n'
                continue
            # Detecting the language of the text
            detected_language = detect_language(temp_body)
            # Adding to the count of all languages
            if detected_language.language not in languages:
                languages[detected_language.language] = 1
            else:
                languages[detected_language.language] += 1
            # Create the file with the text in the corresponding lanugage folder       
            write_to_folder(languages_directory, p, detected_language.language, detected_language.probability, temp_body)
        # Catching exceptions in the log file
        except Exception as e:
            exceptions += str(ExceptionCase(traceback.format_exc(), p)) +"\n"
            number_of_exceptions += 1
    return languages, number_of_exceptions, problematic_paths, exceptions

def lemmatize(files: List[str], root: str, save_path: str, language: str, processor: TextProcessor) -> str:
    long_emails = ""    
    for path in tqdm(files):
        with open(f'{root}/{path}', 'r') as temp_file:
            temp_text = temp_file.read()
        if len(temp_text) >= 1e6:
            # Add to the list of paths that have too long bodies
            long_emails += f'{language}: {path}\n'
            continue
        processed_text = processor.spacy_preprocess(temp_text)
        if not os.path.isdir(f'{save_path}/{language}'):
            # The folder does not exists
            os.mkdir(f'{save_path}/{language}')
        # The folder already exists
        with open(f'{save_path}/{language}/{path}', 'x') as temp_save_file:
            temp_save_file.write(" ".join(processed_text))        
    return long_emails

# endregion

# region LDA

@dataclass()
class ModelValidTopics:
    model: str
    valid: List[int]

@dataclass()
class LdaTopic:
    name: str
    topics_per_model: List[ModelValidTopics]

@dataclass()
class TestConfiguration:
    topics: List[LdaTopic]


def get_lda_results(lda: gensim.models.LdaMulticore, num_words: int = 20) -> typing.Dict:
    answer = {}
    topics = lda.show_topics(num_topics=-1, num_words=num_words)
    for topic_key, compositions in topics:     
        pairs = compositions.split(' + ')
        temp_elements = {}
        for pair in pairs:
            proportion, word = pair.split('*')
            temp_elements[word] = proportion
        answer[f'topic_{topic_key}'] = temp_elements
    return answer

def create_folders_in_path(path:str):
    parts = path.split('/')
    for i, _ in enumerate(parts):
        temp_path = "/".join(parts[:i+1]) # starts at zero and therefore last element does not work
        if not os.path.isdir(temp_path):
            os.mkdir(temp_path)


def recursive_folder_creation(path: str):
    """
    Deprecated (Do not use in scripts newer than August 2022)
    """
    parts = path.split('/')
    for index in range(2, len(parts)):
        new_folder = "/".join(parts[:index])
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)

def create_documents(root: str)-> typing.List[str]:
    answer = []
    for file in os.listdir(root):
        if not os.path.isdir(f'{root}/{file}'):
            with open(f'{root}/{file}', 'r') as temp_file:
                answer.append(temp_file.read())
    return answer

def calculate_coherence(lda: gensim.models.LdaMulticore, bow: BagOfWords, docs: typing.List[typing.List[str]]) -> gensim.models.CoherenceModel:
    cm = gensim.models.CoherenceModel(model=lda,
                    dictionary=bow.dictionary, coherence='c_v',
                    texts=docs)
    return cm

def create_processed_docs(*roots: str, processor: TextProcessor, blacklist: dict, ignore_list: dict, language: str=None) -> List[List[str]]:
    processed_docs = []
    if processor is None:
        processor = TextProcessor(language)
    for root in roots:
        processed_docs.extend([processor.simple_preprocess(doc, blacklist=blacklist, ignore_list=ignore_list) for doc in tqdm(create_documents(root), f'Creating the documents for {root}...')])
    return processed_docs

def create_corpus_from_root(*roots: str, language: str, blacklist: dict, ignore_list: dict) -> Tuple[BagOfWords, List[List[str]]]:
    processor = TextProcessor(language)
    processed_docs = create_processed_docs(*roots, processor=processor, blacklist=blacklist, ignore_list=ignore_list)
    return processor.create_bag_of_words(documents=processed_docs, filter_below=25, filter_above=0.15, keep= 100_000), processed_docs

def run_lda_configurations(min_number_of_topics: int, number_of_topics: int, step: int, corpus: BagOfWords, language: str, processed_docs: List[List[str]], model_case: str, cores: int =1) -> Tuple[dict, dict]:
    results = {}
    print(f'RUNNING MODEL FROM {min_number_of_topics} TOPICS TO {number_of_topics} TOPICS INCREASING BY {step}')
    coherences = {}
    try:
        for i in tqdm(range(min_number_of_topics, number_of_topics, step)):        
            lda_model = gensim.models.LdaMulticore(
                corpus.bow,
                num_topics=i,
                chunksize=10_000, 
                id2word=corpus.dictionary, # mapping of word Id to string value. For debugging purposes
                passes=10,          # How many times it will train through the coprus
                workers=cores          # How many cores will be used for this
            )
            results[f'for_{i}_topics'] = get_lda_results(lda_model, num_words=15)
            recursive_folder_creation(f'saved_models/lda/{model_case}/{language}')
            lda_model.save(f'saved_models/lda/{model_case}/{language}/lda_{i}')
            coherences[i] = calculate_coherence(lda_model, corpus, processed_docs)
    except KeyboardInterrupt:
        print("There was a user interruption, returning available results...")
        return results, coherences
    return results, coherences

def create_bow_from_text(processed_text: List[str], dictionary: gensim.corpora.Dictionary) -> List[Tuple[int, int]]:
    return dictionary.doc2bow(processed_text)

def results_word_counter(text: str, results: str) -> str:
    pat = r'"(.*?)" +'
    words_in_results = Counter(re.findall(pat, results))   
    # Resetting to zero the words
    for key in words_in_results:
        words_in_results[key] -= 1 
    for word in text.split():
        if word in words_in_results:
            words_in_results[word] += 1
    # Converting dictionary to string    
    return " ".join([f'{key}: {value}' for key, value in words_in_results.items()])

def describe_results(text: str, bow: List[Tuple[int, int]], model: gensim.models.LdaMulticore, topics: List[str], num_words: int) -> Tuple[List[str], float]:
    # Get timestamp for text PROCESSING
    processing_time = datetime.now()
    answer = sorted(model[bow], key=lambda tup: -1*tup[1])
    # Get timestamp for lda CLASSIFICATION
    classification_time = datetime.now()
    # Creating results dictionary
    temp_lda_results = []
    for index, score in answer:
        temp_lda_results.append(
            f'{topics[index]} ({index}):\t Score: {score:.3f}\t Words: {results_word_counter(text, model.print_topic(index, num_words))}'
        )
    return temp_lda_results, get_milliseconds_in_between(processing_time, classification_time)
    
def read_configuration(json_config: dict) -> TestConfiguration:
    topics = []
    for topic in json_config["topics"]:
        name = topic["name"]
        valid_model_topics = [ModelValidTopics(tpm["model"], tpm["valid"]) for tpm in topic["topics_per_model"]]
        topics.append(LdaTopic(name, valid_model_topics))
    return TestConfiguration(topics)

def check_if_valid_mail(index: int, config: TestConfiguration, model: str) -> bool:
    # Add to response to which topic the mail belongs to
    for topic in config.topics:
        for model_topics in topic.topics_per_model:
            if model_topics.model != model:
                continue
            elif index in model_topics.valid:
                return True
    return False

# endregion

# region Regex annotation
def create_patterns(config: dict) -> dict:
    patterns = {key: [] for key in config.keys()}
    for key, cases in config.items():
        patterns[key] = []        
        # cases are individual words and combinations of words (dictionary with two keys)
        individual_words = cases["individual"]
        # Preparing the pattern for each individual word
        patterns[key].append(f'({"|".join(individual_words)})')
        combination_words = cases["combination"]
        # Preparing the patterns for combination words        
        for combinations in combination_words:
            patterns[key].append("".join([f'(?=.*{word}*)' for word in combinations.split()]) + ".*")
    return patterns

def is_dict_long_enough(subject: dict, desired_length: int) -> bool:
    for values in subject.values():
        if len(values) < desired_length:
            return False
    return True

def annotate_files(files: List[str], topic_patterns: dict, number_of_cases: int = 400) -> dict:
    annotations = {topic: [] for topic in topic_patterns}
    annotations["Other"] = []
    for file in tqdm(files, "Annotating files..."):
        if is_dict_long_enough(annotations, number_of_cases):
            break
        temp_text = get_relevant_info_mail(file)[0]
        added = False
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if len(re.findall(pattern, temp_text, flags=re.I)) > 0:
                    annotations[topic].append(file)
                    added = True
                    break
        if not added:
            annotations["Other"].append(file)
    return annotations

def check_regex_in_text(text: str, individual: str, combination: List[str] = None) -> bool:    
    """
    This function will receive two texts and a list with patterns to look for.
    The first text is the text to be analyzed. The second text is a regex pattern that will be looked for within the text.
    The list of patterns is a list of combinational words. these combinations must have all words within the text for the result to be True.
    If the regex does not find anything and the combinations of words are also not included in the text, then the result is False.
    """    
    if len(re.findall(individual, text, re.IGNORECASE)) > 0:
        # It found an individual pattern
        return True            
    for combi in combination:
        words = combi.split()
        if all(word in text for word in words):
            # Found all of the words in the combination in the text
            return True
    return False
# endregion

# region miscelaneous
def create_slices(array: List, partitions: int) -> List[List]:
    """Returns a partition of equal or similar sizes of an array

    Args:
        array (List): Any iterable that will be partitioned
        partitions (int): The number of parts that will be created

    Returns:
        List[List]: A list with the parts that were created from the input iterable
    """
    slice_size = len(array) // partitions
    answer = [array[i*slice_size: (i+1)*slice_size] for i in range(partitions-1)]
    answer.append(array[(partitions-1)*slice_size:])
    return answer

def get_real_path(mod: str) -> str:
    mirror = mod.__contains__('virus') or mod.__contains__('spam') or mod.__contains__('clean')
    eap = mod.__contains__('.eml')
    if mirror:
        # prob_1.000_2022-15-04_spam_ea262b297538eb72ad12d3390a843008-sampling-mail.txt        
        elements = mod.split('_')[2:] # ignoring the probability from the name
        return 'mirror/' + '/'.join(elements)[:-4] # ignoring the .txt characters    
    elif not eap:
        # Sample: prob_1.000_2021-10-06_21_2109dd8b255c4fdb28ba2bb7b49ed4a6-currentmail.txt
        elements = mod.split('_')[2:] # ignoring the probability from the name
        return 'eml-samples/' + '/'.join(elements)[:-4] # ignoring the .txt characters    
    else:
        # Sample: prob_0.714_eap_eap_mails_mail_Spearphishing_183.eml.txt
        elements = mod.split('_')[2:] # ignoring the probability form the name
        # eap/eap_mails/{filename - .txt}
        return f'{elements[0]}/{"_".join(elements[1:3])}/{"_".join(elements[3:])[:-4]}'

def get_value_from_nested_list(nested: List) -> float:
    while type(nested) == 'list' or type(nested)=='np.ndarray' or str(type(nested)).__contains__('array'):
        nested = nested[0]    
    return nested

def get_milliseconds_in_between(start: datetime, end: datetime) -> float:
    """
    Gets the amount of milliseconds between two timestapms (datetime) and returns it as a float
    """
    delta = end-start    
    return delta.seconds*1e3 + delta.microseconds/1000

def get_useful_classified_languages(threshold: float, path: str) -> bool:
    """
    This function will detect whether a file is useful or not dependign its probability 
    score for language detection. This values is in its filename.

    Args:
    1. threshold (float): The threshold of confidence the file must beat in order to be valid
    2. path (str): The file path that will be used to check the validity of the file
    """
    file_name = path.split("/")[-1]
    prob = float(file_name.split("_")[0])
    if prob > threshold:
        return True
    else:
        return False

# endregion

# region GRU preprocessing

def create_sequences_from_text(texts: List[str], tokenizer: Tokenizer) -> list:
    return tokenizer.texts_to_sequences(texts)
    
def create_subsequences_from_sequence(sequences: list, labels: List[int] = [], sequence_length: int = 30) -> Tuple[List[int], List[int]]:
    x, y = [], []
    should_retunr_y = True
    if len(labels) == 0:
        labels = list(range(len(sequences)))
        should_retunr_y = False
    for document, label in zip(sequences, labels):
        sub_sequences = np.array([document[i: i+sequence_length] for i, _ in enumerate(document[:-sequence_length])])
        if len(sub_sequences) == 0:
            # The sequence in itself is not long enough, it will be concatenated and padded latter
            sub_sequences = [document]
        y.extend(np.ones(len(sub_sequences))*label)
        x.extend(sub_sequences)
    x, y = pad_sequences(x, sequence_length, padding='post', truncating='pre', value=0), np.array(y)
    if should_retunr_y:
        return x, y
    else:
        return x, None

# Making a proper prediction for the regular input
def full_prediction(input: List[str], tokenizer: Tokenizer, model: Model, sequence_length: int = 30) -> np.array:
    sequences = create_sequences_from_text(input, tokenizer)
    answer = []
    for s in sequences:
        x_tst, _ = create_subsequences_from_sequence([s], sequence_length=sequence_length)
        try:
            y_hat = model.predict(x_tst)
        except Exception as e:
            print(x_tst.shape, e)
            continue
        y_hat = [0 if y <= 0.5 else 1 for y in y_hat]
        y = np.bincount(y_hat)
        if y.max() == y.min():
            # The number of votes is the same. Inconclusive result default to 1
            answer.append(-1)
        else:
            answer.append(y.argmax())
    return np.array(answer)

def balancing_sequences(x : np.array, y: np.array) -> Tuple[np.array, np.array]:
    values, counts = np.unique(y, return_counts=True)
    most, few = max(counts), min(counts)
    if most/few >= 1.5:
        # Needs balancing
        few_value = values[np.argmax(counts)]
        indexes = y == few_value
        sliceable_x, sliceable_y = x[indexes], y[indexes]
        slice_x, slice_y = zip(*random.sample(list(zip(sliceable_x, sliceable_y)), few))
        # Concatenate the new slices to the other labels
        other_indexes = y != few_value
        return np.concatenate((x[other_indexes], slice_x)), np.concatenate((y[other_indexes], slice_y), axis = None)
    else:
        return x, y
# endregion

# region plotting
def plot_confusion_matrix(cf: confusion_matrix, f1: float, name: str):
    ax = plt.subplot()
    sns.heatmap(cf, annot=True, ax=ax, cmap='Blues', fmt='')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.suptitle(f'F1 score: {f1:.5}')
    ax.set_title(f'Confusion Matrix')
    ax.yaxis.set_ticklabels(['No money', 'Money'])
    ax.xaxis.set_ticklabels(['No money', 'Money'])
    plt.savefig(name)
    plt.close()
# endregion
