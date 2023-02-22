import torch


class Dataset:
    def __init__(self, corpus, train_ratio, batch_size, context) -> None:
        self.corpus = corpus
        self.vocab = sorted(list(set(self.corpus)))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        self.encoded_data = self.encode(corpus)

        train_size = int(train_ratio * len(corpus))
        self.train_set = torch.tensor(self.encoded_data[:train_size])
        self.test_set = torch.tensor(self.encoded_data[train_size:])

        self.batch_size = batch_size
        self.context = context

    def sample_batch(self, train=True):
        set_ = self.train_set if train else self.test_set
        idxs = torch.randint(len(set_) - self.context - 1, size=(self.batch_size,))
        x = torch.stack([set_[i : i + self.context] for i in idxs])
        y = torch.stack([set_[i + 1 : i + 1 + self.context] for i in idxs])
        return x, y

    def encode(self, chars):
        return [self.char_to_idx[c] for c in chars]

    def decode(self, idxs):
        return "".join([self.idx_to_char[i] for i in idxs])

    def prepare_input(self, seq):
        return torch.tensor(self.encode(seq)).unsqueeze(0)
