import numpy as np


def broadcast_gradient(grad, other_tensor_shape):
    '''
    Function reshapes gradients to match that of the target tensors
    '''
    ndims_added = grad.ndim - len(other_tensor_shape) # finding extra dimensions
    # Reshaping to match target gradient
    for _ in range(ndims_added): 
        grad = grad.sum(axis=0)
        
    for i, dim in enumerate(other_tensor_shape):
        if dim == 1: 
            grad = grad.sum(axis=i, keepdims=True) 
    return grad

def compute_graph(v, L, visited):
    '''
    Function computes the topological order of nodes in a copmutational graph
    '''
    if v not in visited:
        visited.add(v)
        for child in v.children:
            compute_graph(child, L, visited)
        L.append(v)
    return L

def mse_loss(y_pred, y_true):
    '''
    Function calculates teh mean squared error between two tensors
    '''
    errors = y_pred - y_true
    loss = (errors * errors).sum(keepdims=False)
    return loss