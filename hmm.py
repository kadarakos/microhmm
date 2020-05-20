"""
Hints from https://web.stanford.edu/~jurafsky/slp3/8.pdf
"""

import math
import numpy
from utils import read_train_data
from utils import ngrams
from utils import test_accuracy

class HMM:
    def __init__(self, handle_unks=True, unk_freq=1):

        self.handle_unks = handle_unks
        self.unk_freq = unk_freq
        self.sentences = []
        self.targets = []
        self.tags = []
        self.word_counts = {}
        self.pos_unigram_counts = {}
        self.emission_counts = {}
        self.transition_counts = {}
        self.emission_probs = {}
        self.transition_probs = {}
        self.pos_unigram_probs = {}
        self.word_counts = {}

    def get_word_counts(self, sentences):
        for s in sentences:
            for w in s:
                self.word_counts[w] = self.word_counts.get(w, 0.0) + 1

    def replace_UNK(self, sentences, freq=1):
        """
        Replace the words that appear less than freq times in
        the emission counts.
        """
        #FIXME naive way.
        self.low_freq_words = set([])
        for w, c in self.word_counts.items():
            if c <= freq:
                self.low_freq_words.add(w)
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j] in self.low_freq_words:
                    sentences[i][j] = "<UNK>"

    def compute_counts(self, sentences, targets):
        """
        Load text and estimate parameters for bigram HMM

        :param path: path to data file
        """
        for i in range(len(sentences)):
            # When at the end of a sentence
            # update counts
            words = sentences[i]
            tags = targets[i]
            for i in range(len(tags)):
                self.emission_counts[(words[i], tags[i])] = self.emission_counts.get((words[i], tags[i]), 0.0) + 1
            # Adding start to have estimate for P(t_i|START) usually denote pi
            pos_bigrams = ngrams(['<START>'] + tags, r=2, pad=False)
            for gram in pos_bigrams.values():
                # Update unigram counts
                if len(gram) == 1:
                    self.pos_unigram_counts[gram[0]] = self.pos_unigram_counts.get(gram[0], 0.0) + 1
                else:
                    self.transition_counts[gram] = self.transition_counts.get(gram, 0.0) + 1

    def estimate_params(self):
        """
        Estimate the probabilities for HMM:
        Emission probabilities and transition probabilities as usual.
        """
        for (w, t), c in self.emission_counts.items():
            # P(w_i|t_i) = Count(w_i, t_i) / Count(t_i)
            self.emission_probs[(w, t)] = c / self.pos_unigram_counts[t]
            for t1, t2 in self.transition_counts:
            # P(t_i|t_i-1) = Counts(t_i, t_i-1)/Count(t_i-1)
                self.transition_probs[(t1, t2)] = self.transition_counts[(t1, t2)] / self.pos_unigram_counts[t1]
            Z = sum(self.pos_unigram_counts.values())
            for t in self.pos_unigram_counts:
                self.pos_unigram_probs[t] = self.pos_unigram_counts[t] / Z
        return self.emission_probs, self.transition_probs

    def predict(self, S):
        """
        Viterbi decoding for a single example.
        Given a tokenized sentence S, return tag sequence
        http://www.cs.virginia.edu/~kc2wc/teaching/NLP16/slides/11-Viterbi.pdf
        slide 38
        pseudo code: https://gist.github.com/aaronj1335/9650261
        """
        # Viterbi table is sentence length by number of pos tags
        T = numpy.zeros((len(S), len(self.tags))) - numpy.inf
        backpointers = numpy.zeros((len(S), len(self.tags)))
        # Initialize with log(P(w_1|t_i)) + log(P(t_i|START))
        if self.handle_unks:
            w1 = S[0] if S[0] in self.word_counts else "<UNK>"
        else:
            w1 = S[0]
        for i, tag in enumerate(self.tags):
            pw1_ti = self.emission_probs.get((w1, tag), 1e-8)
            pti_tim1 = self.transition_probs.get(('<START>', tag), 1e-8)
            T[0, i] = math.log(pw1_ti) + math.log(pti_tim1)
        # For each time-step in the sentence.
        for i in range(1, len(S)):
            # For each current tag.
            for t in range(len(self.tags)):
                if self.handle_unks:
                    w = S[i] if S[i] in self.word_counts else "<UNK>"
                else:
                    w = S[i]
                # P(w_i|t_i)
                pw_ti = self.emission_probs.get((w, self.tags[t]), 1e-8)
                # For each previous tag.
                for tm1 in range(len(self.tags)):
                    # P(t_i|t_m1)
                    pti_tim1 = self.transition_probs.get((self.tags[tm1], self.tags[t]), 1e-8)
                    transition = T[i-1, tm1] + math.log(pti_tim1)
                    if transition > T[i, t]:
                        T[i, t] = transition
                        backpointers[i, t] = tm1
                T[i, t] += math.log(pw_ti)
        # Work back in time through the backpointers to get the best sequence
        predicted_tags = [None for x in range(len(S))]
        predicted_tags[0] = self.tags[numpy.argmax(T[0, :])]
        predicted_tags[-1] = self.tags[numpy.argmax(T[-1, :])]
        for i in range(len(S)-2, 0, -1):
            ind = numpy.argmax(T[i, :])
            tag = self.tags[int(backpointers[i+1, ind])]
            predicted_tags[i] = tag
        return predicted_tags

    def train(self, sentences, targets):
        # Get word-counts in the training set
        self.get_word_counts(sentences)
        # Use counts to replace infrequent words with UNK
        if self.handle_unks:
            self.replace_UNK(sentences, self.unk_freq)
        # Get counts to compute transition and emission probs later
        self.compute_counts(sentences, targets)
        # Compute P(w|t) and P(ti|ti-1)
        self.estimate_params()
        self.tags = list(self.pos_unigram_counts.keys())
        self.tags.remove("<START>")

if __name__ == "__main__":
    train_sentences, train_targets = read_train_data('train.txt')
    test_sentences, test_targets = read_train_data('test.txt')
    hmm = HMM(unk_freq=1)
    hmm.train(train_sentences, train_targets)
    test_accuracy(hmm, test_sentences, test_targets)
