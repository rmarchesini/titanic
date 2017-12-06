import numpy as np
from sklearn.utils import shuffle
from utility import getData,cost2,cost,error_rate,softmax,ohe
import matplotlib.pyplot as plt

class ANN(object):
	def __init__(self, M1, M2, M3):
		self.M1 = M1
		self.M2 = M2
		self.M3 = M3

	def genData(self, Nclass = 2000, D = 2):
		X1 = np.random.randn(Nclass, D) + np.array([0, -2])
		X2 = np.random.randn(Nclass, D) + np.array([2, 2])
		X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
		X = np.vstack([X1, X2, X3])

		Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
		return X, Y

	def fit(self, X, Y, learning_rate = 1e-7, iter = 100000, reg=10e-7, show_fig = False, init_param = True):
		#Mezclamos los datos y separamos en los 2 grupos. Entrenamiento y validacion.
		X, Y = shuffle(X, Y)
		Xv, Yv = X[-1000:,:], Y[-1000:]
		X, Y = X[:-1000,:], Y[:-1000]

		#One Hot Encoding
		N, D = X.shape
		K = len(set(Y))
		T = ohe(Y)

		#Inicializacion de parámetros
		if init_param:
			self.W1 = np.random.randn(D, self.M1) / np.sqrt(D)
			self.W2 = np.random.randn(self.M1, self.M2) / np.sqrt(self.M1)
			self.W3 = np.random.randn(self.M2, self.M3) / np.sqrt(self.M2)
			self.W4 = np.random.randn(self.M3, K) / np.sqrt(self.M3)
			self.b1 = np.zeros(self.M1)
			self.b2 = np.zeros(self.M2)
			self.b3 = np.zeros(self.M3)
			self.b4 = np.zeros(K)
		costs=[]

		for i in range(iter):
			#Obtener la prediccion y los valores intermedios.
			pY, Z3, Z2, Z1 = self.forward(X)

			#Actualizar los weights
			##Controlar la derivada de la funcion de activacion
			### Ultima capa
			pY_T = pY - T
			self.W4 -= learning_rate * ( Z3.T.dot(pY_T) + reg*self.W4 )
			self.b4 -= learning_rate * ( pY_T.sum(axis=0) + reg*self.b4 )
			### Tercera capa
			dZ3 = pY_T.dot(self.W4.T) * ( 1 - Z3*Z3 )
			self.W3 -= learning_rate * ( Z2.T.dot(dZ3) + reg*self.W3 )
			self.b3	-= learning_rate * ( dZ3.sum(axis=0) + reg*self.b3 )
			### Segunda capa
			dZ2 = dZ3.dot(self.W3.T) * ( 1 - Z2*Z2 )
			self.W2 -= learning_rate * ( Z1.T.dot(dZ2) + reg*self.W2 )
			self.b2	-= learning_rate * ( dZ2.sum(axis=0) + reg*self.b2 )
			### Primer capa
			dZ1 = dZ2.dot(self.W2.T) * ( 1 - Z1*Z1 )
			self.W1 -= learning_rate * ( X.T.dot(dZ1) + reg*self.W1 )
			self.b1	-= learning_rate * ( dZ1.sum(axis=0) + reg*self.b1 )

			if i % 1000 == 999:
				pYv, _, _, _= self.forward(Xv)
				c = cost2(Yv, pYv)
				costs.append(c)
				e = error_rate(Yv, np.argmax(pYv, axis=1))
				print("i:", i+1, " cost:", c, " error:", e)

		#Graficar el aprendizaje
		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		#Falta agregar la funcion de activacion
		Z1 = np.tanh( X.dot(self.W1) + self.b1 )
		Z2 = np.tanh( Z1.dot(self.W2) + self.b2 )
		Z3 = np.tanh( Z2.dot(self.W3) + self.b3 )
		return softmax( Z3.dot(self.W4) + self.b4), Z3, Z2, Z1

	def predict(self, X):
		#Funcion de activacion
		pY, _, _, _ = forward(X)
		return np.argmax(pY, axis = 1)

	def score(self, X, Y):
		pY = predict(X)
		return 1 - error_rate(Y, np.argmax(pY))