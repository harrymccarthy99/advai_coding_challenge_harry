from src.custom_grad import Tensor
import numpy as np

class Optimizer:
    
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for W, b in self.params.values():
            W.zero_grad()
            b.zero_grad()

    def step(self):
        for W, b in self.params.values():    
            W.data -= self.lr * W.grad.data
            b.data -= self.lr * b.grad.data

class Linear():
    
    def __init__(self, name, row, column, isLastLayer=False):
        self.name = name
        self.row = row
        self.col = column
        self.isLastLayer = isLastLayer

        # Kaiming uniform initialization for weights
        limit = np.sqrt(1.0 / column)  # fan_in = column
        self.W = Tensor(np.random.uniform(-limit, limit, size=(row, column)), requires_grad=True)
        # self.W = Tensor(np.random.normal(0.0, 1.0, size=(row, column)), requires_grad=True)

        self.b = Tensor(0., requires_grad=True)

    def __call__(self, X):
        act = X.matmul(self.W) + self.b
        return act.relu() if not self.isLastLayer else act
        
    def __repr__(self):
        return f"({self.name}): Linear(row={self.row}, column={self.col}, isLastLayer={self.isLastLayer})\n"
        
class NN:

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Linear(f'Linear{i}', sizes[i], sizes[i+1], isLastLayer=(i == len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def __repr__(self):
        s = "model(\n"
        for layer in self.layers:
            s += "  " + str(layer)
        s += ")"
        return s

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'Linear{i}'] = layer.W, layer.b
        return params