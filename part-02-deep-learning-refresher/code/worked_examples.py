"""worked_examples.py - every small number from the lesson, reproduced with plain Python (no libraries needed).

What it does: repeats the by-hand examples from the video one by one: counting parameters, one layer, gradient descent on one
parameter (and a learning rate that is too big), the backpropagation example, softmax and cross-entropy, momentum,
dropout, the warmup-then-cosine schedule and rounding to bfloat16.
What it needs: only Python 3.12 or newer.
What to expect: the numbers printed below, the same ones shown on the slides (see the part README).
Run it: python worked_examples.py
"""
import math
import struct


def parameters():
    """Count the parameters of a 2 -> 16 -> 2 network."""
    first = 2 * 16 + 16                                   # weights and biases of layer 1
    second = 16 * 2 + 2                                   # weights and biases of layer 2
    print(f"parameters: layer 1 = {first}, layer 2 = {second}, total = {first + second}")


def layer():
    """One layer on the input (0.5, -1.0): multiply by weights, add a bias, then bend with ReLU (keep positives only)."""
    x = (0.5, -1.0)
    for name, w, b in (("unit A", (2.0, 1.0), 0.5), ("unit B", (-2.0, 1.0), 0.5)):
        pre = w[0] * x[0] + w[1] * x[1] + b
        print(f"layer, {name}: {w[0]} * {x[0]} + {w[1]} * {x[1]} + {b} = {pre}  ->  ReLU keeps {max(0.0, pre)}")


def descent(lr: float, steps: int = 6):
    """Fit y = 2x from one example (x = 1, y = 2): loss = (w - 2) ** 2, whose gradient is 2 * (w - 2)."""
    w, path = 0.0, [0.0]
    for _ in range(steps):
        w = w - lr * 2 * (w - 2)                          # new parameter = old parameter - learning rate * gradient
        path.append(round(w, 4))
    print(f"gradient descent, learning rate {lr}: {path}")


def backprop():
    """x = 2, w1 = 1, b1 = 1, w2 = 1, b2 = 0, target 6: forward, then the chain rule backwards."""
    x, w1, b1, w2, b2, target = 2.0, 1.0, 1.0, 1.0, 0.0, 6.0
    hidden = max(0.0, w1 * x + b1)
    prediction = w2 * hidden + b2
    loss = (prediction - target) ** 2
    d_pred = 2 * (prediction - target)                    # how the loss reacts to the prediction
    print(f"forward: hidden = {hidden}, prediction = {prediction}, loss = {loss}")
    print(f"backward: dL/dprediction = {d_pred}, dL/db2 = {d_pred}, dL/dw2 = {d_pred * hidden}, "
          f"dL/db1 = {d_pred * w2}, dL/dw1 = {d_pred * w2 * x}")
    nudged = ((w2 + 0.01) * hidden + b2 - target) ** 2
    print(f"nudge w2 by 0.01: loss goes from {loss} to {nudged:.4f} (change {nudged - loss:.4f}, predicted -0.18)")


def softmax_example():
    """Scores 2.0, 1.0, 0.1 become probabilities; the loss is minus the log of the right class's probability."""
    scores = [2.0, 1.0, 0.1]
    exps = [math.exp(s) for s in scores]
    probs = [e / sum(exps) for e in exps]
    print("exp:", [round(e, 3) for e in exps], " sum:", round(sum(exps), 3))
    print("probabilities:", [round(p, 3) for p in probs], " total:", round(sum(probs), 3))
    print(f"cross-entropy for class 0: -ln({probs[0]:.3f}) = {-math.log(probs[0]):.3f}")
    print(f"if the right class were 2: loss = {-math.log(probs[2]):.3f} (a confident miss costs more)")


def momentum(lr: float = 0.1, beta: float = 0.9, steps: int = 4):
    """Gradient descent with momentum on (w - 2) ** 2: buffer = beta * buffer + gradient; w = w - lr * buffer."""
    w, buf, path = 0.0, 0.0, [0.0]
    for _ in range(steps):
        buf = beta * buf + 2 * (w - 2)
        w = w - lr * buf
        path.append(round(w, 3))
    print(f"momentum {beta}, learning rate {lr}: {path}  (it rolls past 2, then comes back)")


def dropout():
    """Dropout with probability 0.5 on four hidden values: keep some, double them, zero the rest."""
    hidden = [2.0, 4.0, 6.0, 8.0]
    mask = [1, 0, 1, 0]                                   # one possible random draw
    scaled = [h * m / (1 - 0.5) for h, m in zip(hidden, mask)]
    print(f"dropout: hidden {hidden}, mask {mask} -> {scaled}; in eval mode the values stay {hidden}")
    print(f"each unit is kept half the time and doubled, so on average it stays the same: {hidden}")


def weight_decay(weight: float = 2.0, lr: float = 0.1, decay: float = 0.1):
    """Weight decay: at every step the weight is multiplied by (1 - lr * decay), a gentle pull towards zero."""
    factor = 1 - lr * decay
    print(f"weight decay: weight {weight}, learning rate {lr}, decay {decay} -> factor {factor:.2f} -> new weight {weight * factor:.2f}")


def schedule(base: float = 0.1, warmup: int = 2, total: int = 10):
    """Learning rate = base * multiplier: a straight climb for `warmup` steps, then half a cosine wave down to 0."""
    rows = []
    for s in range(total + 1):
        mult = (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup)))
        rows.append(round(base * mult, 4))
    print(f"schedule (base {base}, warmup {warmup}, total {total}): {rows}")


def bfloat16(x: float) -> float:
    """Round a float32 number to bfloat16 by keeping its top 16 bits (with round-to-nearest-even)."""
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def precision():
    """bfloat16 keeps float32's range but only about two to three decimal digits."""
    print(f"pi in bfloat16 = {bfloat16(math.pi)}, 1.001 in bfloat16 = {bfloat16(1.001)}, 1000.7 in bfloat16 = {bfloat16(1000.7)}")


if __name__ == "__main__":
    parameters()
    layer()
    descent(0.1)
    descent(1.1)
    backprop()
    softmax_example()
    momentum()
    dropout()
    weight_decay()
    schedule()
    precision()
