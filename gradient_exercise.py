import torch 
from utils import *

def gradient_compute():

    # Define the input tensor
    x = torch.tensor([[1.,2.]], requires_grad=True)
    y = torch.tensor([10.], requires_grad=True)
    A = torch.tensor([[1.,2.0],[2.0,3.0]], requires_grad=True)

    # Compute gradient of f with respect to x
    #grads = torch.autograd.grad(f(x,y), [x,y], create_graph=True)[0]
    # print(grads)
    # print(grads[1])

    # hessian = torch.zeros(x.numel(), x.numel())
    # for i, grad in enumerate(grads):
    #     h = torch.autograd.grad(grad, x, retain_graph=True, grad_outputs=torch.ones_like(grad))
    #     hessian[i] = h[0].view(-1)
    # print(hessian)

    print(quadratic(x,A))

    grads = torch.autograd.grad(quadratic(x,A), x, create_graph=True)
    print(grads)

    hessian = torch.zeros(x.numel(), x.numel())
    for i, grad in enumerate(grads):
        h = torch.autograd.grad(grad, x, retain_graph=True, grad_outputs=torch.ones_like(grad))
        hessian[i] = h[0].view(-1)
    print(hessian)
