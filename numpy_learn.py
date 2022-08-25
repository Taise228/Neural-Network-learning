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
print(f"shape = {a.shape}, dtype = {a.dtype}, dimenstion = {np.ndim(a)}")   #shape is returned in tuple

b = np.array([[1,2,3],[4,5,6]])

print(np.dot(a, b))   # product of matrices a and b

c = np.array([1,1])   #shape = (2,)
d = np.array([[1],[1]])   #shape = (2,1)
print(np.dot(a, c), np.dot(a, d), sep="\n")
#np.dot(a, c) becomes a special operation (matrix * vector). The answer is a vector, multiplied each column of a by c.

def sigmoid(x):
    y = 1 / (1 + np.exp((-1)*x))
    return y

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("sigmoid")
plt.ylim(-0.1, 1.1)
#plt.show()

#branch test