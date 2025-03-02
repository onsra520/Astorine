import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence: str) -> list:
    """
    Split a sentence into words (tokens) using the word_tokenize function from the NLTK library.

    Args:
        sentence (str): sentence to be tokenized.

    Returns:
        list: List of tokens in the sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(word: str) -> str:
    """
    Convert a word to its root form using the Porter Stemmer algorithm.

    Args:
        word (str): The word to be converted.

    Returns:
        str: The root form of the word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence: list, all_words: list) -> np.ndarray:
    """
    Convert a sentence into a numerical vector called a bag of words.

    Args:
        tokenized_sentence (list): List of words (tokens) in the sentence.
        all_words (list): List of all words in the dictionary.

    Returns:
        np.ndarray: Numerical vector (bag of words) representing the sentence.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, words in enumerate(all_words):
        if words in sentence_words:
            bag[idx] = 1
    return bag