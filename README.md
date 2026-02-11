# MaxTorch

A minimal autograd engine and neural network library built from scratch, inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. Created to better understand how automatic differentiation works in PyTorch. I plan on extending this to wrap numpy arrays as Tensors to truly see under the hood of PyTorch in the future but am going to work on other fun projects first :\).

## Features

- **Value**: A scalar value wrapper that tracks gradients
- **Automatic differentiation**: Backpropagation through arbitrary expressions
- **Neural network basic building blocks**: Neuron, Layer, and MLP classes
- **Activation functions**: tanh, sigmoid, ReLU
- **MSE Loss** for regression tasks

## Usage

```python
from Value import Value
from nn import MLP, Loss

model = MLP(3, [4, 1], act="tanh")

xs = [[1, 2, 3], [-1, -2, -3], [1, 2, 4]]
ys = [1.0, -0.8, 1.5]

loss_fn = Loss()
lr = 0.01

for epoch in range(100):
    total_loss = Value(0)
    for x, y in zip(xs, ys):
        pred = model(x)[0]
        total_loss += loss_fn(pred, y)

    model.zero_grad()
    total_loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

print(model([1, 2, 3]))
```

## Files

- `Value.py` - The autograd engine with scalar Value class and backward pass
- `nn.py` - Neural network building blocks (Neuron, Layer, MLP, Loss)

## Acknowledgments

Based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1) video series.
