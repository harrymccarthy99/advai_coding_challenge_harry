from src.utils import compute_graph, broadcast_gradient
import numpy as np

class Tensor:
    """
    Represents a tensor object for automatic differentiation.

    Attributes:
        data (np.ndarray): The data held by the tensor.
        requires_grad (bool): Whether the tensor requires gradient computation.
        children (list): The tensors that contributed to this tensor.
        grad (Tensor): Gradient of the tensor, initialised to zero if requires_grad is True.
        _compute_derivatives (callable): Function to compute derivatives during backpropagation.
    """
    def __init__(self, data, requires_grad = True, children = []):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = requires_grad
        self.children = children
        self.grad = None

        # Change gradients from None to Zero for those that require gradients
        if self.requires_grad:
            self.zero_grad()
        # Stores function.
        self._compute_derivatives = lambda: None
        
    @property
    def shape(self):
        return self.data.shape
  
    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype = float), requires_grad = False)

    def backward(self, grad = None):
        """
        Computes gradients for all tensors in the computation graph.

        Args:
            grad (array-like, optional): Gradient to propagate. If None, defaults to 1 for scalar tensors.
        
        Raises:
            RuntimeError: If a non-scalar tensor has no gradient specified.
        """
        L, visited = [], set()
        tree = compute_graph(self, L, visited)
        if grad is None:
            if self.shape == ():
                self.grad = Tensor(1, requires_grad=False)
            else:
                raise RuntimeError('Backward Pass: non-scaler tensor needs a grad specified')
        else:
            self.grad = Tensor(grad, requires_grad=False) if isinstance(grad, (np.ndarray, list)) else grad
        
        for v in reversed(tree):
            v._compute_derivatives()
  
    #  Reassigning context to magic operators
    def __add__(self, other):
        op = Add(self, other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
    
    def __neg__(self): 
        # -self.
        op = Neg(self)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
        
    def __sub__(self, other):
        # self - other.
        op = Add(self, -other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
        
    def relu(self):
        op = ReLU(self)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
  
    def __matmul__(self, other):
        op = Matmul(self, other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
    
    def __mul__(self, other):
        op = Mul(self, other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result

    def __rmul__(self, other): 
        # other * self.
        op = Mul(other, self)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
  
    def matmul(self, other):
        op = Matmul(self, other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
        
    def sum(self, axis=None, keepdims=True):
        op = Sum(self, axis, keepdims)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
        
    def __truediv__(self, other):
        op = Div(self, other)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result
        
    def __rtruediv__(self, other): 
        # other / self.
        op = Div(other, self)
        result = op.forward()
        result._compute_derivatives = op.compute_derivatives
        return result


class Add():
  def __init__(self, t1, t2):
    self.t1 = t1
    self.t2 = t2
    
  def forward(self):
    self.result = Tensor(np.add(self.t1.data, self.t2.data), children=[self.t1, self.t2])
    return self.result
  
  def compute_derivatives(self):
    if self.t1.requires_grad:
      self.t1.grad.data += broadcast_gradient(self.result.grad.data, self.t1.shape)
      
    if self.t2.requires_grad:
      self.t2.grad.data += broadcast_gradient(self.result.grad.data, self.t2.shape)


class ReLU():
    def __init__(self, t1):
        self.t1 = t1

    def forward(self):
        self.result = Tensor(self.t1.data * (self.t1.data > 0), children=[self.t1])
        return self.result

    def compute_derivatives(self):
        self.t1.grad.data += self.result.grad.data * (self.t1.data > 0)


class Matmul():

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def forward(self):
        self.result = Tensor(self.t1.data @ self.t2.data, children=[self.t1, self.t2])
        return self.result

    def compute_derivatives(self):
      # Check dims
        dim = [i for i in range(len(self.t1.data.shape))]
        if len(dim) > 1:
            dim[-1], dim[-2] = dim[-2], dim[-1]

        if self.t1.requires_grad:
            self.t1.grad.data = self.result.grad.data @ self.t2.data.transpose(dim)
        if self.t2.requires_grad:
            self.t2.grad.data = self.t1.data.transpose(dim) @ self.result.grad.data

class Mul():

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def forward(self):
        self.out = Tensor(np.multiply(self.t1.data, self.t2.data), children=[self.t1, self.t2])
        return self.out

    def compute_derivatives(self):
        if self.t1.requires_grad:
            self.t1.grad.data += broadcast_gradient(self.out.grad.data, self.t2.shape) * self.t2.data
        if self.t2.requires_grad:
            self.t2.grad.data += broadcast_gradient(self.out.grad.data, self.t1.shape) * self.t1.data

class Neg():

    def __init__(self, t1):
        self.t1 = t1
   
    def forward(self):
        self.result = Tensor(-self.t1.data, children=[self.t1])
        return self.result

    def compute_derivatives(self):
        if self.t1.requires_grad:
            self.t1.grad.data += -self.result.grad.data * np.ones_like(self.t1.data)

class Sum():

    def __init__(self, t1, axis=None, keepdims=True):
        self.t1 = t1
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        self.result = Tensor(np.sum(self.t1.data, axis=self.axis, keepdims=self.keepdims), children=[self.t1])
        return self.result

    def compute_derivatives(self):
        if self.axis != None and self.keepdims == False:
            self.t1.grad.data += np.expand_dims(self.result.grad.data, self.axis) * np.ones_like(self.t1.data)
        else:
            self.t1.grad.data += self.result.grad.data * np.ones_like(self.t1.data)


class Div():

    def __init__(self, t1, t2):
        self.t1 = t1 if isinstance(t1, Tensor) else Tensor(t1)
        self.t2 = t2 if isinstance(t2, Tensor) else Tensor(t2)
    
    def forward(self):
        self.result = Tensor(np.divide(self.t1.data, self.t2.data), children=[self.t1, self.t2])
        return self.result

    def compute_derivatives(self):
        if self.t1.requires_grad:
            self.t1.grad.data += broadcast_gradient(self.result.grad.data, self.t1.shape) * 1/(self.t2.data)
        if self.t2.requires_grad:
            self.t2.grad.data += broadcast_gradient(self.result.grad.data, self.t2.shape) * -self.t1.data/(self.t2.data)**2


