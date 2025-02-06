from EZ.word import Word

class Sentence:
    def __init__(self, N, sentence_list):
        self.N = N  # sentence number within the corpus
        self.number_words = len(sentence_list)  # number of words in the sentence
        self.regression_N = 1  # number of inter-word regressions in the sentence
        self.target = 0  # target word number (i.e., N = target word)
        self.max_length = max(sentence_list)
        self.word = [Word(self.max_length) for _ in range(self.number_words)]  # each sentence is a list of words
        self.subj_number = 0

    def add(self, w: Word):
        """Add word to sentence."""
        self.word.append(w)

    def get(self, N: int) -> Word:
        """Access word from sentence."""
        return self.word[N]