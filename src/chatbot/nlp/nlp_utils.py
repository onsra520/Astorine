import sys
import os
import spacy
import torch
import numpy as np
from functools import lru_cache
from typing import List
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    nlp = spacy.load("en_core_web_trf" if torch.cuda.is_available() else "en_core_web_sm")
except OSError:
    print("Could not load en_core_web_trf or en_core_web_sm. Falling back to en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy models: {e}")
    sys.exit(1)

nlp_pool = [nlp]
try:
    if torch.cuda.is_available():
        for _ in range(min(3, os.cpu_count() or 1)):
            try:
                nlp_pool.append(spacy.load("en_core_web_trf"))
            except OSError:
                nlp_pool.append(spacy.load("en_core_web_sm"))
    else:
        for _ in range(os.cpu_count() or 1):
            nlp_pool.append(spacy.load("en_core_web_sm"))
except OSError:
    pass

def process_documents(texts: List[str]) -> List:
    return list(nlp.pipe(texts, batch_size=64))

@lru_cache(maxsize=10000)
def lemmatize_cached(word: str) -> str:
    doc = nlp(word)
    return doc[0].lemma_.lower()

COMMON_LEMMAS = {}
def populate_common_lemmas():
    """Populate dictionary with common English words to avoid repeated computation."""
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she"
        ]
    for word in common_words:
        COMMON_LEMMAS[word] = lemmatize_cached(word)
populate_common_lemmas()

def tokenize_parallel(sentences: List[str]) -> List[List[str]]:
    def process_sentence(args):
        sentence, nlp_instance = args
        doc = nlp_instance(sentence)
        return [token.text for token in doc]

    with ThreadPoolExecutor(max_workers=len(nlp_pool)) as executor:
        work_items = [(sentence, nlp_pool[i % len(nlp_pool)]) for i, sentence in enumerate(sentences)]
        results = list(executor.map(process_sentence, work_items))
    return results

def tokenize(sentence: str) -> list:
    doc = nlp(sentence)
    return [token.text for token in doc]

def stem(word: str) -> str:
    if word in COMMON_LEMMAS:
        return COMMON_LEMMAS[word]
    return lemmatize_cached(word)

def batch_process_stems(words: List[str]) -> List[str]:
    results = []
    batch_words = []
    batch_indices = []
    for i, word in enumerate(words):
        if word in COMMON_LEMMAS:
            results.append(COMMON_LEMMAS[word])
        else:
            batch_words.append(word)
            batch_indices.append(i)
    if batch_words:
        docs = list(nlp.pipe(batch_words, batch_size=64))
        for i, doc in enumerate(docs):
            lemma = doc[0].lemma_.lower()
            original_idx = batch_indices[i]
            results.insert(original_idx, lemma)
            if batch_words[i] not in COMMON_LEMMAS:
                COMMON_LEMMAS[batch_words[i]] = lemma
    
    return results

def create_bow_matrix(tokenized_sentences: List[List[str]], all_words: List[str]) -> np.ndarray:
    word_to_idx = {word: i for i, word in enumerate(all_words)}
    matrix_cpu = np.zeros((len(tokenized_sentences), len(all_words)), dtype=np.float32)
    for i, sentence in enumerate(tokenized_sentences):
        lemmas = set(batch_process_stems(sentence))
        for lemma in lemmas:
            if lemma in word_to_idx:
                matrix_cpu[i, word_to_idx[lemma]] = 1.0
    return matrix_cpu

def bag_of_words(tokenized_sentence: list, all_words: list) -> np.ndarray:
    lemmas = set(batch_process_stems(tokenized_sentence))
    word_to_idx = getattr(bag_of_words, "word_to_idx", None)
    if word_to_idx is None or len(word_to_idx) != len(all_words):
        bag_of_words.word_to_idx = {word: i for i, word in enumerate(all_words)}
        word_to_idx = bag_of_words.word_to_idx
    bag_cpu = np.zeros(len(all_words), dtype=np.float32)
    for lemma in lemmas:
        if lemma in word_to_idx:
            bag_cpu[word_to_idx[lemma]] = 1.0
    
    return bag_cpu