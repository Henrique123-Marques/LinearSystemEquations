#APRESENTACAO - QUESTAO 6 - LISTA 2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração inicial da página
st.set_page_config(page_title="Resolução de Sistemas Lineares", layout="wide")

# Título da apresentação
st.title("📘 Questão 6 - Método de Gauss-Seidel")

# Menu lateral para navegação
st.sidebar.title("🧭 Navegação")
secao = st.sidebar.radio("Escolha a seção:", ["Enunciado", "Metodologia Usada", "Resultados"])

# Seção 1: Enunciado
if secao == "Enunciado":
    st.header("ℹ️ Enunciado")
    st.markdown("""
    O objetivo desta análise é resolver o seguinte sistema linear 3x3 utilizando o sofisticado método iterativo de Gauss-Seidel. 
    Para isso, adotamos como aproximação inicial \( x_0 = (0, 0, 0) \) e estabelecemos uma tolerância de 0,01, 
    garantindo precisão suficiente para as iterações:
    """)
    st.latex(r"""
    \begin{cases}
    5x_1 + x_2 + x_3 = 5 \\
    3x_1 + 4x_2 + x_3 = 6 \\
    3x_1 + 3x_2 + 6x_3 = 0
    \end{cases}
    """)

# Seção 2: Metodologia Usada
elif secao == "Metodologia Usada": 
    
    st.header("""Breve explicativo do método iterativo Guass-Seidel: """)
    st.markdown("""O método iterativo de Gauss-Seidel é uma técnica numérica utilizada para resolver sistemas de equações lineares.
     Desenvolvido no século XIX, ele leva o nome de dois matemáticos alemães: Carl Friedrich Gauss, renomado por suas contribuições 
     em diversas áreas da matemática, e Philipp Ludwig von Seidel, que refinou o conceito iterativo. O método foi formalizado por
      Seidel em 1874, com base em ideias anteriores de Gauss, e serve para encontrar soluções aproximadas de sistemas lineares, 
      especialmente útil em problemas de grande escala ou quando métodos diretos são computacionalmente custosos. Ele funciona
       atualizando iterativamente as variáveis do sistema, usando os valores mais recentes calculados em cada passo, até que a 
       solução convirja para um resultado aceitável dentro de uma tolerância definida.""")

    st.header("⚙️ Metodologia Usada")
    st.markdown("""
    Na resolução desse problema, foi usado a biblioteca NumPy do Python, para manipulação de matrizes e vetores presentes nesse sistema. 
    De modo que fosse possivel organizar o sistema linear em formato matricial, 
    permitindo a aplicação fluida do método de Gauss-Seidel. 

    O processo envolveu a organização dos parâmetros para o problema:
    - **Aproximação inicial**: Definição dos valores iniciais x_0 = (0, 0, 0) como ponto de partida;
    - **Iterações**: Atualização sucessiva das variáveis (x_1, x_2, x_3) para um número máximo definido em 300 iterações;
    - **Critério de parada**: Monitoramento da diferença entre iterações, interrompendo o processo quando a tolerância de 0.01 foi atingida numa condicional.
    """)

    st.markdown("""A partir do método de Gauss-Seidel foi feito um processo iterativo para determinar a solução do sistema linear
     definido pelas matrizes `A` e `b`. Inicialmente, a aproximação inicial `x0`, representada por um vetor de zeros `[0, 0, 0]`, 
     foi copiada para o vetor `x`, que armazenará os valores atualizados das variáveis em cada iteração. O algoritmo então entra
      em um laço que pode se repetir até um número máximo de iterações (`max_iteracoes = 300`), controlando o limite de execução 
      para evitar loops infinitos.

Dentro desse laço, a cada iteração `k`, o vetor `x_antigo` é criado como uma cópia de `x` para preservar os valores da iteração anterior, 
permitindo posteriormente a verificação de convergência. Para cada variável `x[i]` do sistema (onde `i` varia de 0 a `n-1`, com `n`
 sendo o tamanho do sistema), calcula-se uma soma parcial `soma` dos termos que envolvem as demais variáveis `x[j]` (com `j ≠ i`). 
 Essa soma utiliza os valores mais recentes de `x[j]`, característica distintiva do método de Gauss-Seidel, que os atualiza imediatamente
  dentro da mesma iteração. A nova estimativa para `x[i]` é então obtida isolando-a na equação correspondente, usando 
  a fórmula `(b[i] - soma) / A[i,i]`, onde `A[i,i]` é o elemento da diagonal principal da matriz `A`.

Após atualizar todas as variáveis em uma iteração, verifica-se a convergência do método. Isso é feito calculando a diferença
 máxima absoluta entre os valores atuais de `x` e os valores anteriores armazenados em `x_antigo` (`np.max(np.abs(x - x_antigo))`). 
 Se essa diferença for menor que a tolerância pré-definida (`tolerancia = 0.01`), o algoritmo considera que a solução convergiu,
  exibe uma mensagem indicando o número de iterações realizadas (`k+1`) e interrompe o laço com o comando `break`. Caso o número máximo
   de iterações seja atingido sem que a tolerância seja satisfeita, o laço termina naturalmente.

Por fim, os valores da solução aproximada são exibidos com duas casas decimais, apresentando as variáveis `x1`, `x2` e `x3` como 
resultado do processo iterativo. """)

# Seção 3: Resultados
elif secao == "Resultados":
    st.header("📊 Resultados Obtidos")
    st.markdown("""
    Abaixo, exibimos a evolução das soluções aproximadas do método de Gauss-Seidel ao longo das iterações, 
    seguida do resultado final após a convergência.
    """)

    # Definindo o sistema e parâmetros
    A = np.array([[5,1,1],[3,4,1],[3,3,6]], dtype=float)
    b = np.array([5,6,0], dtype=float)
    n = len(b)
    x0 = np.array([0,0,0], dtype=float)
    tolerancia = 0.01
    max_iteracoes = 300

    # Armazenando as soluções aproximadas em cada iteração
    x = x0.copy()
    historico_x = [x.copy()]  # Lista para armazenar os valores de x em cada iteração
    iter_convergencia = max_iteracoes  # Para registrar quando convergiu

    # Método de Gauss-Seidel com histórico
    for k in range(max_iteracoes):
        x_antigo = x.copy()
        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A[i,j] * x[j]
            x[i] = (b[i] - soma) / A[i,i]
        historico_x.append(x.copy())  # Adiciona o novo valor ao histórico
        
        # Verificando convergência
        if np.max(np.abs(x - x_antigo)) < tolerancia:
            iter_convergencia = k + 1
            st.success(f"O sistema convergiu após {iter_convergencia} iterações!")
            break

    # Convertendo o histórico para um array numpy para facilitar o plot
    historico_x = np.array(historico_x)

    # Criando o gráfico de sequências de soluções aproximadas
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(historico_x[:, 0], label=r"$x_1$", marker="o")
    ax.plot(historico_x[:, 1], label=r"$x_2$", marker="s")
    ax.plot(historico_x[:, 2], label=r"$x_3$", marker="^")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Valor")
    ax.set_title("Evolução das Soluções Aproximadas - Método de Gauss-Seidel")
    ax.legend()
    ax.grid(True)

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    # Exibindo a solução final
    st.markdown(f"""
    **Solução Final usando Gauss-Seidel:**  
    \( x_1 = {x[0]:.2f} \)  
    \( x_2 = {x[1]:.2f} \)  
    \( x_3 = {x[2]:.2f} \)
    """)

    st.title('Repositório da Lista 2 📦')
    st.markdown('Github: https://github.com/Henrique123-Marques/LinearSystemEquations')

    st.title('Referências Bibliográficas')
    st.markdown("""
        - GROK. . Disponível em: <https://grok.com/>. 🔗

        - STREAMLIT. Disponível em: <https://docs.streamlit.io/>. 🔗

        - BURDEN, Richard L.; FAIRES, J. Douglas. **Numerical Analysis**. 10. ed. Boston: Cengage Learning, 2016. 

        - CORRÊA. Rejane Izabel Lima.; FREITAS. Rafael de Oliveira.; VAZ. Patricia Machado Sebajos. **Cálculo Númerico**. sagah, 2019. 

""")