import numpy as np

A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]], dtype=float)
b = np.array([7, -8, 6], dtype=float)

# Parâmetros
n = len(b)
x0 = np.array([0.7, -1.6, 0.6])  # Aproximação inicial
tolerancia = 0.01  # 10^-2
max_iteracoes = 300

# Método de Jacobi
x_jacobi = x0.copy()
for k in range(max_iteracoes):
    x_novo = np.zeros(n)
    for i in range(n):
        soma = 0
        for j in range(n):
            if j != i:
                soma += A[i, j] * x_jacobi[j]  # Corrigido: acumula a soma
        x_novo[i] = (b[i] - soma) / A[i, i]
    
    # Verificando a convergência
    if np.max(np.abs(x_novo - x_jacobi)) < tolerancia:
        print(f'O sistema convergiu depois de {k+1} iterações')
        break
    x_jacobi = x_novo.copy()
print(f'Solução pelo Método de Jacobi: x1 = {x_jacobi[0]:.2f}, x2 = {x_jacobi[1]:.2f}, x3 = {x_jacobi[2]:.2f}')