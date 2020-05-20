from utils import read_train_data
from utils import test_accuracy

class FrequencyBaseline:
    """
    Predicts the most frequent tag for each word and
    the most common tag for unknown words.
    """

    def __init__(self):
        self.word_tag_counts = {}
        self.most_frequent_tag = {}
        self.tag_frequencies = {}
        self.top_tag = {}

    def count_occurrences(self, sentences, targets):
        assert len(sentences) == len(targets)
        for s, t in zip(sentences, targets):
            assert len(s) == len(t)
            for word, tag in zip(s, t):
                if word in self.word_tag_counts:
                    if tag in self.word_tag_counts[word]:
                        self.word_tag_counts[word][tag] += 1
                    else:
                        self.word_tag_counts[word][tag] = 1
                else:
                    self.word_tag_counts[word] = {tag: 1}
                self.tag_frequencies[tag] = self.tag_frequencies.get(tag, 0.0) + 1

    def train(self, sentences, targets):
        self.count_occurrences(sentences, targets)
        for word in self.word_tag_counts:
            tag_frequencies = self.word_tag_counts[word]
            most_frequent = sorted(tag_frequencies.items(), key=lambda item: item[1])[-1]
            self.most_frequent_tag[word] = most_frequent[0]
        self.top_tag = sorted(self.tag_frequencies.items(), key=lambda item: item[1])[-1][0]


    def predict(self, sentence):
        predicted_tags = []
        for word in sentence:
            if word in self.most_frequent_tag:
                tag = self.most_frequent_tag[word]
            else:
                tag = self.top_tag
            predicted_tags.append(tag)
        return predicted_tags

if __name__ == "__main__":
    train_sentences, train_targets = read_train_data('train.txt')
    test_sentences, test_targets = read_train_data('test.txt')
    baseline = FrequencyBaseline()
    baseline.train(train_sentences, train_targets)
    test_accuracy(baseline, test_sentences, test_targets)
