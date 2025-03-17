import numpy as np
import matplotlib as plt

A = np.array([[5,1,1],[3,4,1],[3,3,6]], dtype=float)
b = np.array([5,6,0], dtype=float)

#Parametros
n = len(b)
x0 = np.array([0,0,0], dtype=float) #Aproximacao inicial
tolerancia = 0.01
max_iteracoes = 300

#Metodo de Gauss-Seidel
x = x0.copy()
for k in range(max_iteracoes):
	x_antigo = x.copy() #guardar o valor da iteracao anterior
	for i in range(n):
		soma = 0
		for j in range(n):
			if j != i:
				soma += A[i,j] * x[j] #Usa valores atualizados de x
			x[i] = (b[i] - soma) / A[i,i]	

#Verificando convergencia
	if np.max(np.abs(x - x_antigo)) < tolerancia:
		print(f'O sistema convergiu apos {k+1} iterações')
		break

print(f'Solução usando Gauss-Seidel: x1 = {x[0]:.2f}, x2 = {x[1]:.2f}, x3 = {x[2]:.2f}')

#Grafico