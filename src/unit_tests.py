
import numpy as np

# Custom Imports
from src.custom_grad import Tensor

# Torch Imports
import torch


class TestSuite:
    def __init__(self):
        self.tests = []

    def register(self, func):
        """Register a test function."""
        self.tests.append(func)
        return func

    def run_all(self):
        """Run all registered test functions and report results."""
        for test_func in self.tests:
            try:
                test_func()
                print(f"{test_func.__name__} passed!")
            except AssertionError as e:
                print(f"{test_func.__name__} failed: {e}")


# Create a test suite instance
test_suite = TestSuite()

@test_suite.register
def test_add():
    # Custom implementation
    t1 = Tensor([[[1, 2, 3], [1, 2, 3]]], requires_grad=True)
    t2 = Tensor([[[4, 5, 6], [4, 5, 6]]], requires_grad=True)
    t3 = t1 + t2
    t3.backward(Tensor([[[-1, -2, -3], [-1, -2, -3]]], requires_grad=False))
    t1_custom, t2_custom, t3_custom = t1, t2, t3

    # PyTorch implementation
    t1 = torch.Tensor([[[1, 2, 3], [1, 2, 3]]]).requires_grad_()
    t2 = torch.Tensor([[[4, 5, 6], [4, 5, 6]]]).requires_grad_()
    t3 = t1 + t2
    t3.retain_grad()  # Ensure the gradient is retained
    t3.backward(torch.Tensor([[[-1, -2, -3], [-1, -2, -3]]]))
    t1_torch, t2_torch, t3_torch = t1, t2, t3

    # Forward
    assert (t3_custom.data == t3_torch.data.numpy()).all()
    # Backward
    assert (t3_custom.grad.data == t3_torch.grad.data.numpy()).all()
    assert (t2_custom.grad.data == t2_torch.grad.data.numpy()).all()
    assert (t1_custom.grad.data == t1_torch.grad.data.numpy()).all()


@test_suite.register
def test_matmul():
    # Custom implementation
    t1 = Tensor([[[1, 2, 3], [1, 2, 3]]], requires_grad=True)
    t2 = Tensor([[[4, 4], [5, 5], [6, 6]]], requires_grad=True)
    t3 = t1.matmul(t2)
    t3.backward(Tensor([[[-1, -2], [-3, -4]]], requires_grad=False))
    t1_custom, t2_custom, t3_custom = t1, t2, t3

    # PyTorch implementation
    t1 = torch.Tensor([[[1, 2, 3], [1, 2, 3]]]).requires_grad_()
    t2 = torch.Tensor([[[4, 4], [5, 5], [6, 6]]]).requires_grad_()
    t3 = torch.matmul(t1, t2)
    t3.retain_grad()  # Ensure the gradient is retained
    t3.backward(torch.Tensor([[[-1, -2], [-3, -4]]]))
    t1_torch, t2_torch, t3_torch = t1, t2, t3

    # Forward
    assert (t3_custom.data == t3_torch.data.numpy()).all()

    # Backward
    assert (t3_custom.grad.data == t3_torch.grad.data.numpy()).all()
    assert (t2_custom.grad.data == t2_torch.grad.data.numpy()).all()
    assert (t1_custom.grad.data == t1_torch.grad.data.numpy()).all()


@test_suite.register
def test_neg():
    # Custom implementation
    t1 = Tensor([[[1., 2., 3.], [1., 2., 3.]]], requires_grad=True)
    t2 = -t1
    t2.backward(np.ones_like(t1.data))
    t1_custom, t2_custom = t1, t2

    # PyTorch implementation
    t1 = torch.Tensor([[[1., 2., 3.], [1., 2., 3.]]]).requires_grad_()
    t2 = -t1
    t2.retain_grad() 
    t2.backward(torch.ones_like(t1))
    t1_torch, t2_torch = t1, t2

    # Forward
    assert (t2_custom.data == t2_torch.data.numpy()).all()
    # Backward
    assert (t2_custom.grad.data == t2_torch.grad.data.numpy()).all()
    assert (t1_custom.grad.data == t1_torch.grad.data.numpy()).all()

@test_suite.register
def test_sum():
    # Custom Implementation
    t1 = Tensor([[1.,2.,3.], [1.,2.,3.]], requires_grad=True)
    t2 = t1.sum(axis=1, keepdims=False)
    t2.backward(Tensor([10.,20.], requires_grad=False))
    t1_custom, t2_custom = t1, t2

    # Pytorch Implementation 
    t1 = torch.Tensor([[1.,2.,3.],[1.,2.,3.]])
    t1.requires_grad = True
    t2 = t1.sum(axis=1)
    t2.retain_grad()
    t2.backward(torch.Tensor([10.,20.]))
    t1_torch, t2_torch = t1, t2

    # Forward
    assert (t2_custom.data == t2_torch.data.numpy()).all()
    # Backward
    assert (t2_custom.grad.data == t2_torch.grad.data.numpy()).all()
    assert (t1_custom.grad.data == t1_torch.grad.data.numpy()).all()

@test_suite.register
def test_mul():
    # Custom Implementation
    t1 = Tensor([1., 2., 3.], requires_grad=True)
    t2 = Tensor([4., 5., 6.], requires_grad=True)
    t3 = t1 * t2
    t3.backward(Tensor([-1., -2., -3.], requires_grad=False))
    t1_custom, t2_custom, t3_custom = t1, t2, t3

    # Pytorch
    t1 = torch.Tensor([1, 2, 3])
    t1.requires_grad = True
    t2 = torch.Tensor([4, 5, 6])
    t2.requires_grad = True
    t3 = t1 * t2
    t3.retain_grad()
    t3.backward(torch.Tensor([-1, -2, -3]))
    t1_torch, t2_torch, t3_torch = t1, t2, t3

    # Forward
    assert (t3_custom.data == t3_torch.data.numpy()).all()
    # Backward
    assert (t3_custom.grad.data == t3_torch.grad.data.numpy()).all()
    assert (t2_custom.grad.data == t2_torch.grad.data.numpy()).all()
    assert (t1_custom.grad.data == t1_torch.grad.data.numpy()).all()


# Run all tests
test_suite.run_all()