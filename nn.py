from Value import Value
import random

class Neuron():
    def __init__(self, nin):
        self.w = [Value(random.random()) for _ in range(nin)]
        self.b = Value(0)
    
    def __call__(self, x, act = "tanh"):
        summation = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if act == "tanh":
            return summation.tanh()
        elif act == "relu":
            return summation.ReLU()
        elif act == "sigmoid":
            return summation.sigmoid()
        return summation

    def parameters(self):
        return self.w + [self.b]

class Layer():
    def __init__(self, nin, nout):
        self.layer = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x, act="tanh"):
        return [n(x, act) for n in self.layer]
    
    def parameters(self):
        return [p for neuron in self.layer for p in neuron.parameters()]

class MLP():
    def __init__(self, in_channels, hidden_channels, act="tanh"):
        sz = [in_channels] + hidden_channels
        self.act = act
        self.layers = [Layer(sz[idx], sz[idx+1]) for idx in range(len(hidden_channels))]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x, self.act)
        x = self.layers[-1](x, act="none")
        return x
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class Loss():
    def __call__(self, pred, target):
        return (pred-target) **2


if __name__ == "__main__":
    xs = [[1, 2, 3], [-1, -2, -3], [1, 2, 4], [2, 4, 5], 
      [-2, 1, 3], [0, -1, 2], [-1, 3, -2], [3, -2, 1]]
    def f(xs):
        w = [random.random() for _ in range(len(xs[0]))]
        b = random.random()
        ys = []
        for idx, x in enumerate(xs):
            y = sum(wi * xi for wi, xi in zip(w, x)) + b
            y += random.gauss(0, 0.1)
            ys.append(y)
        return ys
    ys = f(xs)
    print(ys)

    model = MLP(3, [4, 4, 1], act="tanh")
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
            p.data -= lr*p.grad
            
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss.data}")
    
    print(f"Input: [-1, -2, -3]: ground truth: {ys[1]}, model prediction: {model([-1, -2, -3])[0].data}")