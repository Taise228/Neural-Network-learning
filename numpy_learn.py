import numpy as np
import matplotlib.pylab as plt

a = np.array([[1,2],[3,4],[5,6]])
print(a)
a += 1   #scalar 1 is broadcast into np.array([[1,1],[1,1],[1,1]])
print(a)
a += np.array([0,1])   #np.array([0,1]) is broadcast into np.array([[0,1],[0,1],[0,1]])
print(a)
a *= 10   #mutiplication is applied to each element
print(a)
a *= np.array([2,1])   #broadcast and multiplication of each element
print(a)
print(f"shape = {a.shape}, dtype = {a.dtype}")

b = np.array([[1,2,3],[4,5,6]])
