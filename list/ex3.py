import numpy as np
#METODO DE JACOBI
print("""
Método de Jacobi

Explicação:
O método de Jacobi é um processo iterativo que resolve Ax = b assumindo uma aproximação inicial para x e atualizando cada componente
 x_i usando os valores da iteração anterior. Ele separa a matriz A em sua diagonal D e o resto R (onde A = D + R), e a
  fórmula iterativa é:
x_i^(k+1) = (b_i - Σ_{j ≠ i} a_{ij} x_j^(k)) / a_{ii}
- k é o número da iteração.
- O método converge se A for diagonalmente dominante (ou seja, |a_{ii}| > Σ_{j ≠ i} |a_{ij}| para cada linha).

Passo a Passo:
1. Escolha uma aproximação inicial para x (ex.: x^(0) = [0, 0, 0]).
2. Para cada iteração k:
   - Calcule x_1^(k+1) = (b_1 - a_{12}x_2^(k) - a_{13}x_3^(k)) / a_{11}.
   - Calcule x_2^(k+1) = (b_2 - a_{21}x_1^(k) - a_{23}x_3^(k)) / a_{22}.
   - Calcule x_3^(k+1) = (b_3 - a_{31}x_1^(k) - a_{32}x_2^(k)) / a_{33}.
3. Verifique a convergência (ex.: diferença entre iterações menor que uma tolerância).
4. Repita até convergir.
""")

#Exemplo abaixo
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
print(f'Solução por Jacobi: x = {x[0]:.4f}, y = {x[1]:.4f}, z = {x[2]:.4f}')

#METODO DE GUASS-SEIDEL
print("""
Método de Gauss-Seidel

Explicação:
O método de Gauss-Seidel é uma melhoria do Jacobi. Em vez de usar os valores da iteração anterior para todos os x_j, 
ele usa os valores já atualizados na mesma iteração para os componentes anteriores. A fórmula é:
x_i^(k+1) = (b_i - Σ_{j < i} a_{ij} x_j^(k+1) - Σ_{j > i} a_{ij} x_j^(k)) / a_{ii}
- Isso geralmente faz com que Gauss-Seidel convirja mais rápido que Jacobi, sob as mesmas condições de convergência 
(matriz diagonalmente dominante).

Passo a Passo:
1. Escolha uma aproximação inicial para x (ex.: x^(0) = [0, 0, 0]).
2. Para cada iteração k:
   - Calcule x_1^(k+1) = (b_1 - a_{12}x_2^(k) - a_{13}x_3^(k)) / a_{11}.
   - Calcule x_2^(k+1) = (b_2 - a_{21}x_1^(k+1) - a_{23}x_3^(k)) / a_{22} (usa o novo x_1).
   - Calcule x_3^(k+1) = (b_3 - a_{31}x_1^(k+1) - a_{32}x_2^(k+1)) / a_{33} (usa os novos x_1 e x_2).
3. Verifique a convergência.
4. Repita até convergir.
""")

for c in range(max_iteracoes):
	x_antigo = x.copy()
	for d in range(n):
		soma= 0
		for e in range(n):
			if e != d:
				soma += A[d, e] * x[e]
		x[d] = (b[d] - soma) / A[d, d]

	#Verificando a convergencia
	if np.max(np.abs(x - x_antigo)) < tolerancia:
		print(f'Convergiu depois de {c+1} iteracoes')
		break

print(f'Solução por Gauss-Seidel: x = {x[0]:.4f}, y = {x[1]:.4f}, z = {x[2]:.4f}')

