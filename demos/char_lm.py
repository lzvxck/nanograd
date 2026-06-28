import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

BUILTIN_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
    "the five boxing wizards jump quickly. "
) * 20

parser = argparse.ArgumentParser()
parser.add_argument("--text-file", default=None)
parser.add_argument("--steps", type=int, default=500)
args = parser.parse_args()

if args.text_file:
    text = open(args.text_file, encoding="utf-8").read()
else:
    text = BUILTIN_TEXT

chars = sorted(set(text))
vocab_size = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
data = [c2i[c] for c in text]

CONTEXT = 3
EMBED_DIM = 16
HIDDEN = 64


class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.fc1 = nn.Linear(CONTEXT * EMBED_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, vocab_size)

    def forward(self, idx):
        x = self.emb(idx).view(idx.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CharLM()
opt = torch.optim.Adam(model.parameters(), lr=0.01)


def make_batch(batch_size=32):
    starts = [random.randint(0, len(data) - CONTEXT - 1) for _ in range(batch_size)]
    xs = torch.tensor([[data[s + k] for k in range(CONTEXT)] for s in starts])
    ys = torch.tensor([data[s + CONTEXT] for s in starts])
    return xs, ys


initial_loss = None
for step in range(args.steps):
    xs, ys = make_batch()
    opt.zero_grad()
    loss = F.cross_entropy(model(xs), ys)
    loss.backward()
    opt.step()
    if step == 0:
        initial_loss = loss.item()
    if (step + 1) % 100 == 0:
        print(f"step {step+1:4d}  loss {loss.item():.4f}")

final_loss = loss.item()
print(f"initial loss: {initial_loss:.4f}  final loss: {final_loss:.4f}")
assert final_loss < initial_loss, "Loss did not decrease"
print("char-lm done.")
