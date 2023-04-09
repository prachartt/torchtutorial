import torch

# Define a scalar-valued function
def f(x,y):
    return torch.sum(x**2)*y

def f2(x,y):
    return x*y

def quadratic(x,A):
    return torch.matmul(x,torch.matmul(A,x.t()))
