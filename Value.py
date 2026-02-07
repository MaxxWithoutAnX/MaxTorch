# TODO:
"""
1. Subtraction
2. Division
3. Exponentiation
4. Activation Functions (Relu, Sigmoid, tanh)
5. Loss functions and NN
6. zero_grad and params
"""

class Value():
    def __init__(self, data, _op="", _children = None):
        self.data = data
        self._op = _op
        self.grad = 0.0
        self._children = _children if _children is not None else set()
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value: {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _op = "+", _children = {self, other})

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op = "*", _children = {self, other})

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
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
    sum = val1 * val1
    prod = val3 * sum
    prod.backward()
    print(val3.grad)
    