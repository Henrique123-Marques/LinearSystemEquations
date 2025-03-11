'''ENUNCIADO: Descreva, fazendo um passo-a-passo explicativo (pode conter um pseudo-código,
fluxograma, etc), como resolver sistemas de equações lineares usando os méto-
dos diretos: (i) de Cramer, (ii) eliminação de Gauss e (iii) decomposição A = LU.'''

print(""" MÉTODOS DIRETOS PARA RESOLVER SISTEMAS DE EQUAÇÕES LINEARES

Sistema de Exemplo:
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3

Matriz aumentada:
A = [[ 2,  1, -1],
     [-3, -1,  2],
     [-2,  1,  2]]
b = [8, -11, -3]""")

#CODIGO REGRA DE CRAMER
import numpy as np

#Explicacao
print("""(i) REGRA DE CRAMER

Explicação:
A Regra de Cramer resolve sistemas lineares usando determinantes. Para um sistema Ax = b:
- Calcula-se o determinante da matriz A (det(A)).
- Para cada variável x_i, substitui-se a i-ésima coluna de A pelo vetor b, calcula-se o determinante da nova matriz (det(A_i)), 
e x_i = det(A_i) / det(A).
- Só funciona se det(A) ≠ 0 (sistema com solução única).

Passo a Passo:
1. Calcule o determinante de A.
2. Para x, substitua a 1ª coluna de A por b e calcule o determinante.
3. Para y, substitua a 2ª coluna de A por b e calcule o determinante.
4. Para z, substitua a 3ª coluna de A por b e calcule o determinante.
5. Divida cada determinante pelo det(A).""")

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
print(""" (ii) ELIMINAÇÃO DE GAUSS

Explicação:
A Eliminação de Gauss transforma a matriz aumentada em uma forma escalonada (triangular superior) usando operações elementares nas linhas. Depois, resolve-se por substituição retroativa.

Passo a Passo:
1. Crie a matriz aumentada [A | b].
2. Use operações de linha para zerar os elementos abaixo da diagonal principal.
3. Resolva o sistema triangular superior por substituição retroativa.""")
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
print("""(iii) DECOMPOSIÇÃO A = LU

Explicação:
A decomposição LU fatora a matriz A em um produto de uma matriz triangular inferior L (com 1s na diagonal) e uma triangular superior U. Resolve-se Ax = b em duas etapas:
1. Ly = b (substituição direta).
2. Ux = y (substituição retroativa).

Passo a Passo:
1. Decomponha A em L e U.
2. Resolva Ly = b para y.
3. Resolva Ux = y para x.""")
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
print("""Comparação e Notas:
- Cramer: Simples, mas computacionalmente caro para sistemas grandes (O(n!)).
- Gauss: Eficiente e amplamente usado, mas pode falhar se houver pivôs nulos sem pivotamento.
- LU: Muito eficiente para resolver múltiplos b com o mesmo A, usado em solvers numéricos.""")