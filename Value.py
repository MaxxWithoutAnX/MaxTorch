# TODO:
"""
1. Subtraction - DONE
2. Division- - DONE
3. Exponentiation - DONE
4. Activation Functions (Relu, Sigmoid, tanh) - DONE
5. Loss functions and NN
6. zero_grad and params
7. Test this and make a working MLP
8. Make it work with Tensors/np arrays then maybe implement softmax and train on MNIST
"""

import math

class Value():
    def __init__(self, data, _children = None):
        self.data = data
        self.grad = 0.0
        self._children = _children if _children is not None else set()
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value: {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children = {self, other})

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children = {self, other})

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out
    
    def __neg__(self):
        out = Value(-self.data, _children=(self,))

        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, other):
        out = Value(self.data ** other, _children = (self,))
        def _backward():
            self.grad += (other) * self.data **(other-1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def exp(self):
        out = Value(math.exp(self.data), _children = {self})
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1/(1+math.exp(-self.data)), _children = {self})
        def _backward():
            self.grad += out.grad * (1-out.data) * out.data
        out._backward = _backward
        return out
    
    def tanh(self):
        out = math.tanh(self.data)
        out = Value(out, _children = {self})

        def _backward():
            self.grad += out.grad * (1-out.data**2)
        out._backward = _backward
        return out
    
    def ReLU(self):
        out = max(0, self.data)
        out = Value(out, _children = {self})

        def _backward():
            if out.data > 0:
                self.grad += 1 * out.grad
            else:
                self.grad = 0
        out._backward = _backward
        return out
    
    def backward(self):
        visited = set()
        topo = []
        def build_topo(v):
            for child in v._children:
                if child not in visited:
                    visited.add(child)
                    build_topo(child)
                topo.append(child)
        build_topo(self)
        topo.append(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    

if __name__ == "__main__":
    val1 = Value(5)
    val2 = Value(3)
    val3 = Value(9)
    expo = val3.sigmoid()
    expo.backward()
    print(val3.grad)
    print(expo)
    