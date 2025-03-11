import numpy as np
#METODO DE JACOBI

#EXPLICATIVO
print("""""")

#Exemplo

A = np.array([[4,-1,1], [2,5,2], [1,2,4]], dtype=float)
b = np.array([8, 3, 11], dtype=float)

#Parametros
n = len(b)
x = np.zeros(n) #Aprox inicial
x_novo = np.zeros(n)
tolerancia = 1e-6
max_iteracoes = 100

for k in range(max_iteracoes):
	for i in range(n):
		soma = 0
		for j in range(n):
			if j != i:
				soma += A[i,j] * x[j]
		x_novo[i] = (b[i] - soma) / A[i,i]

	#Verificando a convergencia
	if np.max(np.abs(x_novo - x)) < tolerancia:
		print(f'Convergiu apos {k+1} iteracoes')
		break

	x = x_novo.copy()
print(f'Solução por Jacobi: x = {x[0]:.6f}, y = {x[1]:.6f}, z = {x[2]:.6f}')