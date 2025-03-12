import numpy as np
#Sistema linear
A = np.array([[5,2,1], [-1,4,2], [2,-3,10]], dtype=float)
b = np.array([7,3,-1], dtype=float)

#Parametros
n = len(b)
x0 = np.array([-2.4, 5, 0.3]) #Aproximacao inicial
tolerancia = 0.01 #10^-2
max_iteracoes = 100 

#Metodo de Jacobi
x_jacobi = x0.copy()
for k in range(max_iteracoes):
	x_novo = np.zeros(n)
	for i in range(n):
		soma = 0
		for j in range(n):
			if j != i:
				soma += A[i, j] * x_jacobi[j]
		x_novo[i] = (b[i] - soma) / A[i,i]

	#Verificação da convergencia
	if np.max(np.abs(x_novo - x_jacobi)) < tolerancia:
		print(f'O sistema convergiu após {k+1} iterações')
		break
	x_jacobi = x_novo.copy()
print(f'Solução pelo Método de Jacobi: x1 = {x_jacobi[0]:.4f}, x2 = {x_jacobi[1]:.4f}, x3 = {x_jacobi[2]:.4f}')