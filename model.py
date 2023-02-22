import torch.nn as nn
import torch
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size, context, embedding_dim=384) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)
        self.context = context

    def forward(self, x):
        return self.embedding_layer(x)

    @torch.no_grad()
    def generate(self, s, max_tokens):
        for _ in range(max_tokens):
            logits = self(s[:, -self.context :])  # B,T,C
            next_word_logits = logits[:, -1, :]  # B,C
            probs = F.softmax(next_word_logits, dim=-1)
            predicted_next_word = torch.multinomial(probs, num_samples=1)
            s = torch.cat((s, predicted_next_word), dim=1)
        return s
