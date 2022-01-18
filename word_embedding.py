import numpy as np
class word_embedding_provider:

    def __init__(self):
        self.unks = []

    def read_glove_vecs(self, glove_file):
        with open(glove_file, 'r', encoding="utf8") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split(' ')
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            i = 1
            words_to_index = {}
            index_to_words = {}
            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1
        return words_to_index, index_to_words, word_to_vec_map

    # This function will act as a "last resort" in order to try and find the word
    # in the words embedding layer. It will basically eliminate contiguously occuring
    # instances of a similar character
    def __cleared(self, word):
        res = ""
        prev = None
        for char in word:
            if char == prev: continue
            prev = char
            res += char
        return res

    def __sentence_to_indices(self, sentence_words, word_to_index, max_len, i):
        global X
        for j, w in enumerate(sentence_words):
            try:
                index = word_to_index[w]
            except:
                w = self.__cleared(w)
                try:
                    index = word_to_index[w]
                except:
                    index = word_to_index['unk']
                    self.unks.append(w)
            self.X[i, j] = index

    def get_glove_word_embedding(self, word_to_index, index_to_word, word_to_vec_map, clean_desc, max_len):
        self.X = np.zeros((len(clean_desc), max_len))  # Data Padding
        for i, tokens in enumerate(clean_desc):
            self.__sentence_to_indices(tokens, word_to_index, max_len, i)
        return self.X