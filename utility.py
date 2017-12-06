import numpy as np

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

def getData(data = 'fer2013.csv'):
	X = []
	Y = []
	first = True
	for line in open(data):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(pix) for pix in row[1].split()])
	X, Y = np.array(X) / 255.0, np.array(Y)
	return X, Y

def relu(x):
	return x * (x > 0)


def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def error_rate(T, Y):
	return np.mean(T != Y)

def ohe(Y):
	K=len(set(Y))
	N=len(Y)
	T=np.zeros((N,K))
	for i in range(N):
		T[i,Y[i]] = 1
	return T
