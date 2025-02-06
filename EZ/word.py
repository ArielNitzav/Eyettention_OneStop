class DV:
    def __init__(self, max_length):
        # Fixation-duration measures
        self.FFD = 0  # first-fixation duration (ms)
        self.GD = 0  # gaze duration (ms)
        self.GoPast = 0  # go-past time (ms)
        self.SFD = 0  # single-fixation duration
        self.TT = 0  # total time (ms)

        # Fixation-probability measures
        self.Pr1 = 0  # probability of making single fixation
        self.Pr2 = 0  # probability of making 2+ fixations
        self.PrF = 0  # probability of fixating word (during first or second pass)
        self.PrS = 0  # probability of skipping word

        # Distributions
        self.distSFD = [0] * max_length  # SFD distributions (i.e., IOVPs)
        self.distPr1 = [0] * max_length  # Pr1 distributions (i.e., first-fixation landing sites)
        self.distPr2 = [0] * max_length  # Pr2 distributions (i.e., refixation probabilities)
        self.NFirstPassFixations = 0  # number of first-pass fixations
        self.NFixations = 0  # total number of fixations
        self.NSFD = [0] * max_length  # counters for SFD distributions

        self.position1 = 0

        self.NRegIn = 0  # # regressions in
        self.NRegOut = 0  # # first-pass regressions out
        self.NRegOutFull = 0  # # regressions out
        self.FirstPassFFD = 0  # # first-pass first fixation duration
        self.FirstPassGD = 0  # # first-pass gaze duration
        self.FirstFixProg = 0  # first-fixation progression


class IV:
    def __init__(self):
        self.cloze = 0  # cloze predictability
        self.frequencyClass = 0  # frequency class (1 = 1-10, 2 = 11-100, etc.)
        self.frequency = 0  # frequency count (per million)
        self.length = 0  # number of letters in word
        self.letters = ""  # word's spelling
        self.log10Frequency = 0  # log10(frequency)
        self.N = 0  # number of word within sentence
        self.OVP = 0  # optimal-viewing position
        self.position0 = 0  # cumulative number of the space preceding the word (i.e., leftmost edge of space)
        self.position1 = 0  # cumulative number of first letter (i.e., leftmost edge of letter)
        self.positionN = 0  # cumulative number of last letter (i.e., rightmost edge of letter)


class Word:
    def __init__(self, max_length):
        self.dv = DV(max_length)  # initialize word DVs
        self.iv = IV()  # initialize word IVs


def word_DVs(text, trace): # wordDVs(ArrayList<Sentence> text, ArrayList<Fixation> trace)
    """Convert trace into word-based dependent variables (DVs).
        every scanpath -> updates word level measures
    
    """
    for i in range(text.number_words):
        w = Word(len(text.get(i).dv.distSFD))  # Initialize word class for DVs

        # Count total number of fixations:
        w.dv.NFixations = sum(1 for fixation in trace if fixation.word == i)

        # Count number of first-pass fixations:
        for j, fixation in enumerate(trace):
            if fixation.word == i:
                post_regression = any(prev.word > i for prev in trace[:j])
                previously_fixated = any(prev.word == i for prev in trace[:j-1])
                if not post_regression and (not previously_fixated or trace[j - 1].word == i):
                    w.dv.NFirstPassFixations += 1
                    if w.dv.NFirstPassFixations == 1:
                        w.dv.FirstPassFFD = fixation.duration
                    w.dv.FirstPassGD += fixation.duration
                    if j < len(trace) - 1 and trace[j + 1].word < i:
                        w.dv.NRegOut += 1
                    if w.dv.NFirstPassFixations == 1:
                        w.dv.FirstFixProg = 1

        # Determine value and position of SFD:
        if w.dv.NFirstPassFixations == 1 and w.dv.NFixations == 1:
            for fixation in trace:
                if fixation.word == i:
                    w.dv.SFD = fixation.duration
                    pos = int(fixation.position - text.get(fixation.word).iv.position0)
                    w.dv.distSFD[pos] = w.dv.SFD
                    w.dv.NSFD[pos] += 1

        # Determine value and position of FFD, and if a refixation occurred:
        for fixation in trace:
            if fixation.word == i and w.dv.FFD == 0:
                w.dv.FFD = fixation.duration
                pos = int(fixation.position - text.get(fixation.word).iv.position0)
                w.dv.distPr1[pos] += 1
                if trace.index(fixation) < len(trace) - 1 and trace[trace.index(fixation) + 1].word == i:
                    w.dv.distPr2[pos] += 1

        # Calculate GD:
        in_first_run, started_first_run = True, False
        if w.dv.NFirstPassFixations > 0:
            for fixation in trace:
                if fixation.word == i:
                    started_first_run = True
                    if in_first_run:
                        w.dv.GD += fixation.duration
                elif started_first_run:
                    in_first_run = False

        # Calculate TT:
        w.dv.TT = sum(fixation.duration for fixation in trace if fixation.word == i)

        # Calculate GoPast:
        in_fixation = next((j for j, fixation in enumerate(trace) if fixation.word == i), -1)
        out_fixation = next((j for j, fixation in enumerate(trace) if fixation.word > i), -1)
        if in_fixation > -1 and out_fixation > -1:
            w.dv.GoPast = sum(trace[j].duration for j in range(in_fixation, out_fixation))

        # Calculate NRegOutFull:
        w.dv.NRegOutFull = sum(1 for j in range(len(trace) - 1) if trace[j].word == i and trace[j + 1].word < i)

        # Calculate NRegIn:
        w.dv.NRegIn = sum(1 for j in range(1, len(trace)) if trace[j].word == i and trace[j - 1].word > i)

        # Update counters for fixation-probability measures:
        if w.dv.NFirstPassFixations == 0:
            w.dv.PrS += 1
        elif w.dv.NFirstPassFixations == 1:
            w.dv.Pr1 += 1
        else:
            w.dv.Pr2 += 1
        if w.dv.NFixations > 0:
            w.dv.PrF += 1

        # Add values to running totals used to calculate means across subjects:
        text.get(i).dv.SFD += w.dv.SFD
        text.get(i).dv.FFD += w.dv.FFD
        text.get(i).dv.GD += w.dv.GD
        text.get(i).dv.TT += w.dv.TT
        text.get(i).dv.GoPast += w.dv.GoPast
        text.get(i).dv.Pr1 += w.dv.Pr1
        text.get(i).dv.Pr2 += w.dv.Pr2
        text.get(i).dv.PrS += w.dv.PrS
        text.get(i).dv.PrF += w.dv.PrF
        text.get(i).dv.NFirstPassFixations += w.dv.NFirstPassFixations
        text.get(i).dv.NFixations += w.dv.NFixations
        for j in range(len(w.dv.distPr1)):
            text.get(i).dv.distPr1[j] += w.dv.distPr1[j]
            text.get(i).dv.distPr2[j] += w.dv.distPr2[j]
            text.get(i).dv.distSFD[j] += w.dv.distSFD[j]
            text.get(i).dv.NSFD[j] += w.dv.NSFD[j]
        text.get(i).dv.NRegIn += w.dv.NRegIn
        text.get(i).dv.NRegOut += w.dv.NRegOut
        text.get(i).dv.NRegOutFull += w.dv.NRegOutFull
        text.get(i).dv.FirstPassGD += w.dv.FirstPassGD
        text.get(i).dv.FirstPassFFD += w.dv.FirstPassFFD
        text.get(i).dv.FirstFixProg += w.dv.FirstFixProg
    
    return text

def word_means(text, includeRegressionTrials=True):

    """Calculate word-based means.
    NSubjects - > num of scanpaths
    maxLength -> longest word
    text -> list of paragraphs
    NSentences - > len of list of paragraphs #339vscode-remote://ssh-remote%2Blaccl-srv1.dds.technion.ac.il/data/home/ariel.kr/Eyettention/EZ_output.ipynb
    
    """
    for word in text.word:
        word_dv = word.dv

        if word_dv.Pr1 > 0:
            word_dv.SFD /= word_dv.Pr1

        if word_dv.PrF > 0:
            word_dv.TT /= word_dv.PrF
            word_dv.GoPast /= word_dv.PrF
            word_dv.FFD /= word_dv.PrF
            word_dv.GD /= word_dv.PrF

        if (word_dv.Pr1 + word_dv.Pr2) > 0:
            for k in range(text.max_length):
                if word_dv.distPr1[k] > 0:
                    word_dv.distPr2[k] /= word_dv.distPr1[k]
                word_dv.distPr1[k] /= (word_dv.Pr1 + word_dv.Pr2)
                if word_dv.NSFD[k] > 0:
                    word_dv.distSFD[k] /= word_dv.NSFD[k]
            
            word_dv.FirstPassGD /= (word_dv.Pr1 + word_dv.Pr2)
            word_dv.FirstPassFFD /= (word_dv.Pr1 + word_dv.Pr2)

        if includeRegressionTrials:
            text.regressionN = 0

        word_dv.Pr1 /= (text.subj_number - text.regressionN)
        word_dv.Pr2 /= (text.subj_number - text.regressionN)
        word_dv.PrS /= (text.subj_number - text.regressionN)
        word_dv.PrF /= (text.subj_number - text.regressionN)

        word_dv.NFixations /= text.subj_number
        word_dv.NRegIn /= text.subj_number
        word_dv.NRegOut /= text.subj_number
        word_dv.NRegOutFull /= text.subj_number
        word_dv.FirstFixProg /= text.subj_number

        word.dv = word_dv