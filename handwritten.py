"""
Handwritten Character Recognition with a CNN (PyTorch).

Datasets : MNIST (digits 0-9) or EMNIST (letters / digits / mixed characters), downloaded automatically.
Model    : Convolutional Neural Network with batch norm, dropout and data augmentation.
Image processing : invert to white-on-black, crop, scale to 20 px, centre on a 28x28 canvas (MNIST style).
Extension: word recognition by splitting a line of separated characters and classifying each one.
           For connected (cursive) handwriting, the next step is a CRNN with CTC loss (see the end of the file).

Usage
  python handwriting.py train --dataset mnist --epochs 6               # digits, about 99% accuracy
  python handwriting.py train --dataset emnist-letters --epochs 10     # letters A-Z
  python handwriting.py train --dataset emnist-balanced --epochs 10    # digits + letters (47 classes)
  python handwriting.py app --dataset mnist                            # draw with the mouse and recognise
  python handwriting.py predict --image digit.png --dataset mnist      # recognise an image file
  python handwriting.py predict --image word.png --dataset emnist-letters --word   # a word of separated letters

Datasets: mnist | emnist-digits | emnist-letters | emnist-balanced | emnist-byclass
Requires: torch torchvision numpy pillow   (tkinter comes with the standard Python installer on Windows)
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MEAN, STD = 0.1307, 0.3081
MODEL_DIR = Path("models")
LABELS = {
    "mnist": list("0123456789"),
    "emnist-digits": list("0123456789"),
    "emnist-letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "emnist-balanced": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"),
    "emnist-byclass": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
}


# ----------------------------------------------------------------------------- #
# Image processing (works on any photo / scan / drawing)
# ----------------------------------------------------------------------------- #
def to_white_on_black(img):
    """PIL image -> float array 0..255 where the ink is bright and the background is dark."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if np.median(a) > 127:  # dark ink on light paper: invert
        a = 255.0 - a
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max() * 255.0
    return a


def normalize_glyph(a):
    """Crop to the ink, scale the longest side to 20 px, centre by centre of mass on 28x28.
    Returns a normalised tensor of shape (1, 1, 28, 28), or None if the image has no ink."""
    import torch

    mask = a > 40
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(size, Image.LANCZOS), dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small

    total = canvas.sum()
    if total > 0:  # move the centre of mass to the middle, like MNIST digits
        cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
        cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
        canvas = np.roll(canvas, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    x = torch.from_numpy(canvas / 255.0).float().unsqueeze(0).unsqueeze(0)
    return (x - MEAN) / STD


def preprocess(img):
    return normalize_glyph(to_white_on_black(img))


def segment_characters(a, min_width=2):
    """Split a line of separated characters using the vertical ink projection.
    Returns a list of (x_start, x_end) column ranges."""
    cols = (a > 40).any(axis=0)
    spans, start = [], None
    for x, ink in enumerate(cols):
        if ink and start is None:
            start = x
        elif not ink and start is not None:
            spans.append([start, x])
            start = None
    if start is not None:
        spans.append([start, len(cols)])
    return [(s, e) for s, e in spans if e - s >= min_width]


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
def build_model(num_classes):
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

    return nn.Sequential(
        *block(1, 32), *block(32, 32), nn.MaxPool2d(2), nn.Dropout(0.25),    # 28 -> 14
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2), nn.Dropout(0.25),   # 14 -> 7
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def _transpose(x):  # EMNIST images are stored transposed compared to MNIST
    return x.transpose(1, 2)


def _minus_one(y):  # EMNIST 'letters' labels run from 1 to 26
    return y - 1


def get_dataset(name, train):
    from torchvision import datasets, transforms as T

    if name not in LABELS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {', '.join(LABELS)}")
    steps = [T.ToTensor()]
    if name.startswith("emnist"):
        steps.append(T.Lambda(_transpose))
    if train:
        steps.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    steps.append(T.Normalize((MEAN,), (STD,)))
    tf = T.Compose(steps)

    if name == "mnist":
        ds = datasets.MNIST("data", train=train, download=True, transform=tf)
    else:
        split = name.split("-", 1)[1]
        ds = datasets.EMNIST("data", split=split, train=train, download=True, transform=tf,
                             target_transform=T.Lambda(_minus_one) if split == "letters" else None)
    return ds, LABELS[name]


# ----------------------------------------------------------------------------- #
# Train / evaluate
# ----------------------------------------------------------------------------- #
def evaluate(model, loader, dev):
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / total


def cmd_train(a):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", dev)
    train_ds, labels = get_dataset(a.dataset, True)
    test_ds, _ = get_dataset(a.dataset, False)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1000, num_workers=0)
    print(f"{a.dataset}: {len(train_ds)} training images, {len(test_ds)} test images, {len(labels)} classes")

    model = build_model(len(labels)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs // 3), gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{a.dataset}.pt"

    best = 0.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
            n += len(x)
        sched.step()
        acc = evaluate(model, test_dl, dev)
        print(f"Epoch {epoch}/{a.epochs}  loss {run / n:.4f}  test accuracy {acc * 100:.2f}%")
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(), "labels": labels, "dataset": a.dataset}, path)
    print(f"Best test accuracy {best * 100:.2f}%. Model saved to {path}")


# ----------------------------------------------------------------------------- #
# Prediction
# ----------------------------------------------------------------------------- #
def load_model(a):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(a.model) if a.model else MODEL_DIR / f"{a.dataset}.pt"
    if not path.exists():
        raise SystemExit(f"Model file {path} not found. Train it first:\n"
                         f"  python handwriting.py train --dataset {a.dataset}")
    ck = torch.load(path, map_location=dev)
    model = build_model(len(ck["labels"])).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["labels"], dev


def classify(model, x, labels, dev, k=3):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        p = F.softmax(model(x.to(dev)), dim=1)[0].cpu()
    top = p.topk(min(k, len(labels)))
    return [(labels[int(i)], float(v)) for v, i in zip(top.values, top.indices)]


def predict_word(model, labels, dev, img, space=0.5):
    """Recognise a line of separated characters. A gap wider than `space` x the typical character width
    becomes a space. Returns (text, mean confidence)."""
    a = to_white_on_black(img)
    spans = segment_characters(a)
    if not spans:
        return "", 0.0
    widths = [e - s for s, e in spans]
    space_gap = space * float(np.median(widths))
    text, confs = "", []
    for i, (s, e) in enumerate(spans):
        if i > 0 and s - spans[i - 1][1] > space_gap:
            text += " "
        x = normalize_glyph(a[:, s:e])
        if x is None:
            continue
        ch, conf = classify(model, x, labels, dev, k=1)[0]
        text += ch
        confs.append(conf)
    return text, float(np.mean(confs)) if confs else 0.0


def cmd_predict(a):
    model, labels, dev = load_model(a)
    img = Image.open(a.image)
    if a.word:
        text, conf = predict_word(model, labels, dev, img, a.space)
        print(f"Recognised text: '{text}'  (average confidence {conf * 100:.1f}%)")
        return
    x = preprocess(img)
    if x is None:
        raise SystemExit("No handwriting found in the image.")
    for ch, p in classify(model, x, labels, dev):
        print(f"  {ch}   {p * 100:5.1f}%")


# ----------------------------------------------------------------------------- #
# Drawing app (Tkinter)
# ----------------------------------------------------------------------------- #
def cmd_app(a):
    import tkinter as tk
    from PIL import ImageDraw

    model, labels, dev = load_model(a)
    W, H, PEN = 560, 280, 18
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    root = tk.Tk()
    root.title(f"Handwritten Character Recognition ({a.dataset})")
    canvas = tk.Canvas(root, width=W, height=H, bg="black", cursor="cross")
    canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
    big = tk.Label(root, text="?", font=("Segoe UI", 64, "bold"), width=6)
    big.grid(row=1, column=0, rowspan=2)
    detail = tk.Label(root, text="Draw a character with the mouse", justify="left", font=("Segoe UI", 12))
    detail.grid(row=1, column=1, sticky="w")
    word_mode = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Word mode (write separated letters)", variable=word_mode).grid(row=2, column=1, sticky="w")
    last = [None]

    def predict():
        if not img.getbbox():
            return
        if word_mode.get():
            text, conf = predict_word(model, labels, dev, img)
            big.config(text=text or "?", font=("Segoe UI", 32 if len(text) > 6 else 48, "bold"))
            detail.config(text=f"Average confidence: {conf * 100:.1f}%")
        else:
            x = preprocess(img)
            if x is None:
                return
            res = classify(model, x, labels, dev)
            big.config(text=res[0][0], font=("Segoe UI", 64, "bold"))
            detail.config(text="\n".join(f"{c}   {p * 100:.1f}%" for c, p in res))

    def press(e):
        last[0] = (e.x, e.y)
        r = PEN // 2
        canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
        draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def move(e):
        if last[0] is None:
            return
        canvas.create_line(*last[0], e.x, e.y, fill="white", width=PEN, capstyle=tk.ROUND)
        draw.line([*last[0], e.x, e.y], fill=255, width=PEN)
        r = PEN // 2
        draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
        last[0] = (e.x, e.y)

    def release(e):
        last[0] = None
        predict()

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, W, H], fill=0)
        big.config(text="?", font=("Segoe UI", 64, "bold"))
        detail.config(text="Draw a character with the mouse")

    canvas.bind("<Button-1>", press)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", release)
    tk.Button(root, text="Clear", width=10, command=clear).grid(row=1, column=2, padx=10)
    tk.Button(root, text="Recognise", width=10, command=predict).grid(row=2, column=2, padx=10)
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description="Handwritten character recognition with a CNN")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--dataset", default="mnist", choices=list(LABELS))
        s.add_argument("--model", default=None, help="path to a saved model (default models/<dataset>.pt)")

    s = sub.add_parser("train", help="train the CNN")
    common(s)
    s.add_argument("--epochs", type=int, default=6)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--lr", type=float, default=1e-3)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("predict", help="recognise an image file")
    common(s)
    s.add_argument("--image", required=True)
    s.add_argument("--word", action="store_true", help="split into characters and read a whole word")
    s.add_argument("--space", type=float, default=0.5, help="word mode: gap (x character width) that counts as a space")
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("app", help="draw with the mouse and recognise")
    common(s)
    s.set_defaults(fn=cmd_app)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #
# Going further: full words and sentences with a CRNN
#   Image (1 x 32 x W) -> CNN feature extractor -> Bi-LSTM over the width -> Linear(num_chars + 1)
#   Train with torch.nn.CTCLoss on a word/line dataset such as IAM Handwriting or synthetic text images,
#   then decode with greedy CTC decoding. This handles connected (cursive) writing that the
#   segmentation-based --word mode above cannot.
# --------------------------------------------------------------------------- 