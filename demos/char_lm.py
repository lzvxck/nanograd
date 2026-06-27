import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from nanograd import Tensor
from nanograd.nn import Embedding, Linear
from nanograd.optim import Adam


def main():
    parser = argparse.ArgumentParser(description="Character-level LM demo")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = (
            "to be or not to be that is the question "
            "whether tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune "
            "or to take arms against a sea of troubles " * 20
        )

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    context_len = 3
    embed_dim = 16
    hidden = 64

    np.random.seed(42)
    emb = Embedding(vocab_size, embed_dim)
    lin1 = Linear(embed_dim * context_len, hidden)
    lin2 = Linear(hidden, vocab_size)
    params = [*emb.parameters(), *lin1.parameters(), *lin2.parameters()]
    opt = Adam(params, lr=args.lr)

    def get_batch(size=32):
        ix = np.random.randint(context_len, len(data) - 1, size=size)
        xb = np.stack([data[i - context_len:i] for i in ix])
        yb = data[ix]
        return xb, yb

    losses = []
    for step in range(1, args.steps + 1):
        xb, yb = get_batch()
        batch_size = xb.shape[0]

        # Embedding lookup then reshape — Tensor.reshape tracks grad correctly
        flat_emb = emb(xb.flatten())  # (B*context_len, embed_dim)
        inp = flat_emb.reshape(batch_size, context_len * embed_dim)

        logits = lin2(lin1(inp).relu())
        loss = logits.cross_entropy(yb)

        for p in params:
            p.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  loss={loss.data:.4f}")

    assert losses[-1] < losses[0], (
        f"char-LM loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )
    print(f"Done. Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
