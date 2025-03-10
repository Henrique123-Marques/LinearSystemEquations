'''ENUNCIADO: Descreva, fazendo um passo-a-passo explicativo (pode conter um pseudo-código,
fluxograma, etc), como resolver sistemas de equações lineares usando os méto-
dos diretos: (i) de Cramer, (ii) eliminação de Gauss e (iii) decomposição A = LU.'''

#CODIGO REGRA DE CRAMER
import numpy as np

#Matria A e vetor b
A = np.array([[2,1,-1], [-3,-1,2], [-2,1,2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

#Determinante inicial de A
det_A = np.linalg.det(A)

#O sistema tem solucao?
if det_A == 0:
	print('Sistema nao tem solucao unica (det_(A) = 0)')
else:
	#Matrizes para as variaveis x, y, z
	A_x = A.copy()
	A_x[:, 0] = b #Substituicao da 1 coluna por b
	A_y = A.copy()
	A_y[:, 1] = b #Substituicao da segunda coluna por b
	A_z = A.copy() 
	A_z[:, 2] = b #Substituicao da terceira coluna por b

	#Calculo dos novos determinantes
	det_A_x = np.linalg.det(A_x)
	det_A_y = np.linalg.det(A_y)
	det_A_z = np.linalg.det(A_z)

	#Solucao
	x = det_A_x / det_A
	y = det_A_y / det_A
	z = det_A_z / det_A

	print(f'Solucao: x = {x:.2f}, y = {y:.2f} e z = {z:.2f}')


#CODIGO ELIMINACAO DE GAUSS
#Criacao da Matriz aumentada
n = len(b)
aumentada = np.hstack((A, b.reshape(-1,1)))

#Metodo eliminacao de Gauss
for i in range(n):
	pivot = aumentada[i, i]
	if pivot == 0:
		print('Pivot Zero detectado, sistema nao tem solucao unica')
		break

		#Nomalizacao da linha do pivo
		aumentada[i] = aumentada[i] / pivot

		#Eliminar elementos abaixo do pivo
		for j in range(i + 1, n):
			fator = aumentada[j,i]
			aumentada[j] -= fator * aumentada[i]

#Substituicao Rotrativa
x = np.zeros(n)
for i in range(n - 1, -1, -1):
	x[i] = aumentada[i, -1] - np.dot(aumentada[i, i + 1:n], x[i + 1:n])
print(f'Solucao: x = {x[0]:.2f}, y= {x[1]:.2f}, z = {x[2]:.2f}')

#CODIGO DECOMPOSICAO A = LU, L = LOWER AND U = UPPER
from scipy.linalg import lu
#Decomposicao LU
P, L, U = lu(A) #P é a matriz de permutacao (pivotamento)

#Resolver Ly = Pb (com pivotamento)
b_permutado = np.dot(P,b) #Ajustar o vetor b conforme a permutacao
y = np.zeros_like(b)
for i in range(len(b)):
	y[i] = b_permutado[i] - np.dot(L[i, :i], y[:i])

x = np.zeros_like(b)
for i in range(len(b) -1, -1, -1):
	x[i] = (y[i] - np.dot(U[i,i + 1:], x[i + 1:])) / U[i, i]

print(f'Solucao: x = {x[0]:.2f}, y = {x[1]:.2f}, z = {x[2]:.2f}')