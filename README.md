# TextProcessor
This text processor was created to handle basic text operations used in most NLP projects. The goal of building this package is having reusable code in one central location instead of repeating the calls in multiple projects. 

The processing logic is build on top of <a href="https://spacy.io">spacy</a>. It was chosen because it supports the languages we require (Spanish, German, English and French) while having the best performance. We also tried previously with <a href="https://stanfordnlp.github.io/stanza/">Stanza</a> but this library takes considerably longer to perform the same text processing tasks we use in this package.

## Contents of the package
The package consist of two `dataclasses` <a href="https://docs.python.org/3/library/dataclasses.html">(See documentation for more info)</a>. These are:
1. TextProcessor: The actual processing tool that is build using the `spacy tool`. This class is the one that has the functions to process the text.
2. BagOfWords: This `dataclass` is simpler and with a more specific purpose. One of the functions of the `TextProcessor` will generate a `BagOfWords` as a result. This function us used to build a corpus and a `gensim Dictionary`. This comes in handy when using the `LDA` models of `gensim`. Both of these variables are what make a `BagOfWords dataclass`.

## How to use?

In order to instantiate a `text_processor` tool, it is required to specify the language using the first part of the <a href="http://www.lingoes.net/en/translator/langcode.htm">ISO format</a> value. i.e. for English language the tool should be instantiated as:

```
from text_processor.text_processor import TextProcessor
processor = TextProcessor('en')
```

The processing processes available in this module are:
1. Simple processing: This operation takes a `string` and returns a `List` of `strings`. This list is made of the words inside the input that remain after the simple filter from the `spacy` tool. This filter gets rid of stopwords in the specified language and using extra files specified by the user, it can ignore documents that contain words in the `blaklist` input or ignore words included in the `ignore_list` input. (check documentation of functions for more detail).
2. Lemmatizytion: It takes processed text and applies <a href="https://en.wikipedia.org/wiki/Lemmatisation">lemmatization</a> to it in order to get the most reliable input to any NLP model.
3. Create a bag of words: This function will using several documents to create a corpus and a `gensim dictionary` that will be necessary to use the LDA models in the `gensim` package.


## License
<a href="https://choosealicense.com/licenses/mit/">MIT</a>
