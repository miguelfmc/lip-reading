"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

PyTorch toy examples
"""

import numpy as np
import torch


x = torch.empty(5, 3)
print(x)

X = torch.rand(2, 5, 5, 3)
print(X)

Y = X.numpy()
print(type(Y))

X = torch.tensor([[2, 3], [4, 5]])
beta = torch.tensor([1, -1])
# z = X @ beta
z = torch.matmul(X, beta)
print(z)

vec = np.random.randn(1000, 10)
tens = torch.from_numpy(vec)
print(tens[:5, :5])

print(torch.cuda.current_device())