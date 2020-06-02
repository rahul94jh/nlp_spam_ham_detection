import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Optional


""" run this if getting error for wordnet """
# import nltk
# nltk.download("wordnet")

## initialise the inbuilt Stemmer and the Lemmatizer
stemmer = PorterStemmer()
word_lemmatizer = WordNetLemmatizer()


def preprocess(document: str, stem: Optional[bool] = True) -> str:
    """[summary]
        changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence
    Arguments:
        document {str} -- message to be processed

    Keyword Arguments:
        stem {Optional[bool]} -- apply stemming if true else apply lemmatization {DEfault : True}
    """
    # change sentence to lower case
    document = document.lower()
    # tokenize into words
    words = word_tokenize(document)
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    if stem:
        words = [stemmer.stem(word) for word in words]  # apply stemming if stem is True
    else:
        words = [
            word_lemmatizer.lemmatize(word, pos="v") for word in words
        ]  # apply lemmatization if not stem

    # join words to make sentence
    documents = " ".join(words)

    return documents


def filter_message(data_set: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """[summary]
        Filters the message using the preprocess function and length of word in message.
    Arguments:
        data_set {List[Tuple[str, str]]} -- list of tuple containing message and corresponding label
    """

    messages_set = []
    for (message, label) in data_set:
        words_filtered = [
            word.lower()
            for word in preprocess(message, stem=False).split()
            if len(word) >= 3
        ]
        messages_set.append((words_filtered, label))

    return messages_set


def get_words_in_messages(messages: List[str]) -> List[str]:
    """[summary]
        creating a single list of all words in the entire dataset for bag of words creation.
    Arguments:
        messages {List[str]} -- list containing messages
    """
    all_words = []
    for words in messages:
        all_words.extend(words)
    return all_words


def create_bag_of_words(messages: List[str]) -> List[str]:
    """[summary]
        Creating a bag of words list using an intuitive FreqDist, to eliminate all the duplicate words.
        Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.
    Arguments:
        messages {List[str]} -- list containg messages
    """
    all_words = get_words_in_messages(messages)
    wordlist = nltk.FreqDist(all_words)
    bag_of_words = wordlist.keys()
    return bag_of_words
