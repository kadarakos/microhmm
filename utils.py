import tqdm

def ngrams(s, r, pad=False):
    """
    Takes a list as input and returns all n-grams up to a certain range.

    :param s: input list
    :param r: range
    :param pad: whether to return ngrams containing a "PAD" token
    :return: dicttionary {id: ngram}
    """
    # add padding
    S = s + ["<PAD>"] * (r - 1)
    l = len(S)
    ng = {}
    curr_id = 0
    seen = set()
    for i in range(l-1):
        for j in range(i, i+r):
            gram = tuple(S[i:j+1])
            elements = set(gram)
            # Don't store ngrams with PAD tokens
            if "<PAD>" in elements and not pad:
                continue
            # Don't store ngrams with only the pad token
            elif len(elements) == 1 and "<PAD>" in elements:
                continue
            elif gram not in seen:
                seen.add(gram)
                ng[curr_id] = gram
                curr_id += 1
    return ng

def read_train_data(path):
    sentences = []
    targets = []
    with open(path, 'r') as f:
        words = []
        tags = []
        for line in f:
            if line == "\n":
                if len(tags) != len(words):
                    raise
                sentences.append(words)
                targets.append(tags)
                words = []
                tags = []
            else:
                word, pos, chunk = line.split()
                words.append(word)
                tags.append(chunk)
    return sentences, targets

def test_accuracy(model, sentences, targets):
    num_tokens = 0
    correct = 0
    for i, s in tqdm.tqdm(enumerate(sentences)):
        tags = model.predict(s)
        for j, t in enumerate(tags):
            num_tokens += 1.0
            correct += int(targets[i][j] == t)
    print(correct / num_tokens)