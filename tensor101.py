#This file teaches tensor stuff to torch beginners

import torch
import numpy as np

#Defining Tensor

t1 = torch.tensor([1.2,3], dtype=float)
print(t1.dtype)

#You can also define tensor using numpyarray

nparray=np.array([1,2,3])

t2 = torch.tensor(nparray)

#You can also change the type of any tensor
print(t2)
t2=t2.float()
print(t2)


#Reshaping

'''Reshaping a tensor involves changing its dimensions while 
keeping the same number of elements. PyTorch provides the view() 
method to reshape tensors.'''

#1. 
x = torch.randn(2, 3)

#x has two rows and three columns

print(x)
x.shape

#means converting x to 2 rows of size 1x3
y = x.view(2, 1, 3)
print(y)

#2. Flattening a tensor:
#You can flatten a tensor into a 1D tensor using view() as follows:
z=y.view(-1)
print(z)
#the -1 automatically calculates the number of dimensions of the flattened vector
z=x.view(-1,2)
print(z)

'''Slicing:  slicing is the operation of selecting a subset of elements from a tensor.
You can use slicing to extract a part of a tensor or modify specific elements of a tensor. '''

#1: Slicing a tensor by indices

x = torch.randn(3, 4)
#y below is a 1-d tensor
y = x[1, :]  #This will give you the second row of the tensor


#2: Slicing a tensor by range:

x = torch.randn(3, 4)
y = x[0:2, :] 

#3: Slicing a tensor by step:

x = torch.randn(3, 4)
y = x[0, 0::2]

#y is a tensor of shape (2,) containing alternate elements of first row of x

#4. Advanced Slicing

x = torch.randn(3, 4)

mask = x > 0  #boolean tensor 

y = x[mask]

print(mask)

print(y)  #a 1-d tensor with only positive elements


#Tensor Concatenation

'''concatenation is the operation of joining two or more tensors 
together along a specified dimension. Concatenation is a useful operation 
for combining data from different sources or 
for increasing the dimensionality of a tensor.'''

#1. Concatenation tensors along existing dimensions

x = torch.randn(3, 4)
y = torch.randn(3, 4)
z = torch.cat((x, y), dim=0)

#2. Concatenating tensors along a new dimension:

x = torch.randn(3, 4)
y = torch.randn(3, 4)
z = torch.cat((x, y), dim=1)

print(x)
print(y)
print(z)

#Matrix Multiplication

#1. Element wise multiplications

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a * b
print(c)

#2. Matrix Multiplication

a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.matmul(a, b)
print(c)


#Batch Multiplication

a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c = torch.matmul(a, b)

print(a)
print(b)
print(c)





















#debug line




#Tensor Operations

# 1. addition
# 2. multiplication



