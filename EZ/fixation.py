class Fixation:
    def __init__(self, duration, number, word):
        self.duration = duration  # duration in milliseconds
        self.number = number  # word number being fixated (i.e., 1 - N)
        self.position = 0.0  # cumulative within-sentence character position of fixation
        self.word = word  # cumulative within-sentence word number being fixated
