import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    #### Hàm tokenize dùng để tách một câu thành các từ hoặc token.
        - Token là một đơn vị nhỏ nhất của văn bản được tách ra từ một câu hoặc đoạn văn.
        - Token có thể là một từ, một dấu câu, một số, hoặc thậm chí là một ký tự đặc biệt.
        - Tokenization là quá trình tách một câu hoặc một đoạn văn thành các token.

    #### Ví dụ:
    Câu gốc: "ah F*ck, I'm very handsome."
    → ['ah', 'F', '*', 'ck', ',', 'I', "'m", 'very', 'handsome', '.']
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    #### Hàm stem dùng để để thực hiện stemming trên một từ
        - Stemming là quá trình chuyển đổi một từ về dạng gốc (root form) của nó
        - Stemming thường được sử dụng để xử lý các từ giống nhau nhưng có thể được viết dưới dạng khác nhau

    #### Cách thức hoạt động:
    Thuật toán stemming (như Porter Stemmer) sử dụng các quy tắc heuristic để loại bỏ các hậu tố (suffixes)
    của từ, từ đó tìm ra dạng gốc (root form).

        - Loại bỏ hậu tố số nhiều:
            -  Ví dụ: "cats" → "cat", "dogs" → "dog".
        - Loại bỏ hậu tố động từ:
            - Ví dụ: "running" → "run", "jumped" → "jump".
        - Loại bỏ hậu tố tính từ:
            - Ví dụ: "happier" → "happy", "biggest" → "big".
        - Áp dụng các quy tắc đặc biệt:
            - Ví dụ: "organization" → "organ", "national" → "nation".
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    #### Hàm bag_of_words dùng để chuyển một câu thành một vector số gọi là bag of words
        - Bag of words là một vector số mà mỗi phần tử trong vector biểu diễn cho một từ trong từ điển.
        - Giá trị của phần tử là số lần xuất hiện của từ trong câu.

    #### Ví dụ:
    sentence = ["thank", "you", "I", "am", "so", "cool"]

    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

    bog   = [  0,     1,     1,     1,     0,      1,      1]

    """
    # Chuyển đổi mỗi từ về dạng gốc (root form)
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Với những từ không nằm trong từ điển, sẽ là 1
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag