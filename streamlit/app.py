#APRESENTACAO - QUESTAO 6 - LISTA 2
import streamlit as st
import numpy as np

# Configuração inicial da página
st.set_page_config(page_title="Resolução de Sistemas Lineares", layout="wide")

# Título da apresentação
st.title("📘 Questão 6 - Método de Gauss-Seidel")

# Menu lateral para navegação
st.sidebar.title("🧭 Navegação")
secao = st.sidebar.radio("Escolha a seção:", ["Enunciado", "Metodologia Usada", "Resultados"])
st.sidebar.markdown('Desenvolvido por Henrique Marques Santos Silva;')

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

elif secao == "Resultados":
    st.header("📊 Resultados Obtidos")
    st.markdown("""
    Abaixo, exibimos a evolução das soluções aproximadas do método de Gauss-Seidel ao longo das iterações, 
    seguida do resultado final após a convergência.
    """)

    # Importando Plotly
    import plotly.graph_objects as go

    # Exibindo o código de definição do sistema
    st.markdown("### Código de Definição do Sistema")
    st.code("""
    # Definindo o sistema e parâmetros
    A = np.array([[5,1,1],[3,4,1],[3,3,6]], dtype=float)
    b = np.array([5,6,0], dtype=float)
    n = len(b)
    x0 = np.array([0,0,0], dtype=float)
    tolerancia = 0.01
    max_iteracoes = 300
    """, language="python")

    # Exibindo o código do método de Gauss-Seidel
    st.markdown("### Implementação do Método de Gauss-Seidel")
    st.code("""
    x = x0.copy()
    historia_x = [x.copy()]  # Lista para armazenar a evolução de x
    for k in range(max_iteracoes):
        x_antigo = x.copy() # Guardar o valor da iteração anterior
        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A[i,j] * x[j] # Usa valores atualizados de x
            x[i] = (b[i] - soma) / A[i,i]   
        historia_x.append(x.copy())  # Armazena o novo x
        # Verificando convergência
        if np.max(np.abs(x - x_antigo)) < tolerancia:
            print(f'O sistema convergiu após {k+1} iterações')
            break
    """, language="python")

    # Executando o código para mostrar os resultados
    A = np.array([[5,1,1],[3,4,1],[3,3,6]], dtype=float)
    b = np.array([5,6,0], dtype=float)

    # Parâmetros
    n = len(b)
    x0 = np.array([0,0,0], dtype=float)  # Aproximação inicial
    tolerancia = 0.01
    max_iteracoes = 300

    # Método de Gauss-Seidel com histórico
    x = x0.copy()
    historia_x = [x.copy()]  # Lista para armazenar a evolução de x
    for k in range(max_iteracoes):
        x_antigo = x.copy()  # Guardar o valor da iteração anterior
        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A[i,j] * x[j]  # Usa valores atualizados de x
            x[i] = (b[i] - soma) / A[i,i]   
        historia_x.append(x.copy())  # Armazena o novo x
        # Verificando convergência
        if np.max(np.abs(x - x_antigo)) < tolerancia:
            st.write(f'O sistema convergiu após {k+1} iterações')
            break

    # Exibindo a solução final
    st.markdown(f"""
    **Solução Final usando Gauss-Seidel:**  
    \( x_1 = {x[0]:.2f} \)  
    \( x_2 = {x[1]:.2f} \)  
    \( x_3 = {x[2]:.2f} \)
    """)

    # Criando o gráfico com Plotly
    iteracoes = list(range(len(historia_x)))
    x1_vals = [x[0] for x in historia_x]
    x2_vals = [x[1] for x in historia_x]
    x3_vals = [x[2] for x in historia_x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iteracoes, y=x1_vals, mode='lines+markers', name='x₁'))
    fig.add_trace(go.Scatter(x=iteracoes, y=x2_vals, mode='lines+markers', name='x₂'))
    fig.add_trace(go.Scatter(x=iteracoes, y=x3_vals, mode='lines+markers', name='x₃'))
    fig.update_layout(
        title="Evolução das Soluções Aproximadas (Gauss-Seidel)",
        xaxis_title="Iterações",
        yaxis_title="Valores de x",
        template="plotly_white"
    )

    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    st.title('Repositório da Lista 2 📦')
    st.markdown('Github: https://github.com/Henrique123-Marques/LinearSystemEquations')

    st.title('Referências Bibliográficas 📘')
    st.markdown("""
        - GROK. Disponível em: <https://grok.com/>. 🔗
        - STREAMLIT. Disponível em: <https://docs.streamlit.io/>. 🔗
        - PLOTLY. Disponível em: <https://plotly.com/python/>. 🔗
        - BURDEN, Richard L.; FAIRES, J. Douglas. **Numerical Analysis**. 10. ed. Boston: Cengage Learning, 2016. 
        - CORRÊA, Rejane Izabel Lima.; FREITAS, Rafael de Oliveira.; VAZ, Patricia Machado Sebajos. **Cálculo Númerico**. sagah, 2019. 
    """)