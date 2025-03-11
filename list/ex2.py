import numpy as np

#Sistema Linear 3x3
A = np.array([[4, -1 , 1], [2, 5, 2], [1, 2, 4]], dtype = float)
b = np.array([8, 3, 11], dtype = float)

#Metodo de Cramer
det_A = np.linalg.det(A)

if det_A == 0:
	print('O sistema nao tem solucao unica (det_A(0) = 0)')
else:
	#Matrizes para as variaveis x,y e z
	A_x = A.copy()
	A_x[:, 0] = b #Subistituicao da primeira coluna por b
	A_y = A.copy()
	A_y[:, 1] = b #Subistituicao da segunda coluna por b
	A_z = A.copy()
	A_z[:, 2] = b #Subistituicao da terceira coluna por b

	#Novos determinantes
	det_A_x = np.linalg.det(A_x)
	det_A_y = np.linalg.det(A_y)
	det_A_z = np.linalg.det(A_z)
	#Solucoes
	x = det_A_x / det_A
	y = det_A_y / det_A
	z = det_A_z / det_A
	print(f'Solução do ex2: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}')

#ELIMINACAO DE GAUSS
n = len(b)
matriz_aumentada = np.hstack((A, b.reshape(-1,1)))

for i in range(n):
	pivo = matriz_aumentada[i,i]
	if abs(pivo) < 1e-10:
		print('Pivo Zero encontrado, sistema nao possui solucao unica')
		break

	#Normalizacao da linha do pivo
	matriz_aumentada[i] = matriz_aumentada[i] / pivo
	#Eliminacao dos elementos abaixo do pivo
	for j in range(i + 1, n):
		fator = matriz_aumentada[j,i]
		matriz_aumentada[j] -= fator * matriz_aumentada[i]

#Substituicao rotrativa
x = np.zeros(n)
for i in range(n - 1, -1, -1):
	x[i] = matriz_aumentada[i, -1]
	for j in range(i + 1, n):
		x[i] -= matriz_aumentada[i,j] * x[j]
print(f'Solução por metodo de Eliminacao de Gauss: x = {x[0]:.2f}, y = {x[1]:.2f}, z = {x[2]:.2f}')

#METODO DE DECOMPOSICAO A = LU
from scipy.linalg import lu
P, L, U = lu(A) #P = matriz de permutacao (pivotamento)

b_permutado = np.dot(P,b) #Ajuste do vetor conforme a permutacao
y = np.zeros_like(b)
for i in range(len(b)):
	y[i] = b_permutado[i] - np.dot(L[i, :i], y[:i])

x = np.zeros_like(b)
for i in range(len(b) -1, -1, -1):
	x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
print(f'Solução pelo metodo de Decomposicao A=LU: x = {x[0]:.2f}, y = {x[1]:.2f}, z = {x[2]:.2f}')


