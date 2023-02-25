# %%
from dataset import Dataset
from model import BigramModel
import torch
import torch.nn as nn
from tqdm import tqdm

CONTEXT = 8
TRAIN_RATIO = 0.9
EMBEDDING_DIM = 384

NUM_ITERS = 25000
EVAL_EVERY = 1000
EVAL_ITERS = 200

BATCH_SIZE = 32


@torch.no_grad()
def eval(model, dataset, loss_fn):
    for set_ in ["train", "test"]:
        total_loss = 0
        for i in range(EVAL_ITERS):
            is_train = set_ == "train"
            x, y = dataset.sample_batch(train=is_train)
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            total_loss += loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))
        print(f"{set_.capitalize()} Loss: {total_loss/EVAL_ITERS:.04}")


def train(model, dataset: Dataset, loss_fn, optim, num_iters, eval_every):
    model = model.cuda()
    for i in tqdm(range(num_iters), leave=True):
        if i % eval_every == 0 or i == num_iters - 1:
            eval(model, dataset, loss_fn)

        x, y = dataset.sample_batch(train=True)
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()


def inspect_model_generation_quality(model, dataset):
    start_seq = dataset.prepare_input(["\n"])
    generated_seq = model.generate(start_seq, max_tokens=10000)
    decoded_seq = dataset.decode(generated_seq.numpy().tolist()[0])
    print(decoded_seq)


def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    dataset = Dataset(
        corpus=text, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE, context=CONTEXT
    )

    model = BigramModel(
        vocab_size=len(dataset.vocab),
        context=dataset.context,
        embedding_dim=EMBEDDING_DIM,
    )

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    train(model, dataset, loss_fn, optim, NUM_ITERS, EVAL_EVERY)

    inspect_model_generation_quality(model.cpu(), dataset)


if __name__ == "__main__":
    main()
