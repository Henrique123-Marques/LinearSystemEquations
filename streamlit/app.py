#APRESENTACAO - QUESTAO 6 - LISTA 2
import streamlit as st

# Configuração inicial da página
st.set_page_config(page_title="Resolução de Sistemas Lineares", layout="wide")

# Título da apresentação
st.title("Questão 6 - Método de Gauss-Seidel")

# Menu lateral para navegação
st.sidebar.title("Navegação")
secao = st.sidebar.radio("Escolha a seção:", ["Enunciado", "Metodologia Usada", "Resultados"])

# Seção 1: Enunciado
if secao == "Enunciado":
    st.header("Enunciado")
    st.markdown("""
    O objetivo desta análise é resolver o seguinte sistema linear 3x3 utilizando o sofisticado método iterativo de Gauss-Seidel. 
    Para isso, adotamos como aproximação inicial x_0 = (0, 0, 0) e estabelecemos uma tolerância de 0,01, 
    garantindo precisão suficiente para as iterações:
    """)
    
    # Sistema linear em LaTeX
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

    st.header("Metodologia Usada")
    st.markdown("""
    Na resolução desse problema, foi usado a biblioteca NumPy do Python, para manipulação de matrizes e vetores presentes nesse sistema. 
    De modo que fosse possivel organizar o sistema linear em formato matricial, 
    permitindo a aplicação fluida do método de Gauss-Seidel. 

    O processo envolveu a organização dos parâmetros para o problema:
    - **Aproximação inicial**: Definição dos valores iniciais x_0 = (0, 0, 0) como ponto de partida;
    - **Iterações**: Atualização sucessiva das variáveis (x_1, x_2, x_3) para um número máximo definido em 300 iterações;
    - **Critério de parada**: Monitoramento da diferença entre iterações, interrompendo o processo quando a tolerância de 0.01 foi atingida numa condicional.
    """)

    st.markdown("Para o método de Gauss-Seidel...")

# Seção 3: Resultados
elif secao == "Resultados":
    st.header("Resultados Obtidos")
    st.markdown("""
    Após a aplicação rigorosa do método de Gauss-Seidel, os resultados emergiram como uma solução refinada para o sistema proposto. 
    Abaixo, apresentamos os valores aproximados das variáveis (x_1, x_2, x_3), obtidos com base nas iterações realizadas:

    - **Convergência**: O algoritmo demonstrou comportamento estável, alcançando a precisão desejada dentro do limite de tolerância.
    - **Valores finais**: [Aqui seriam inseridos os valores numéricos finais, acompanhados de uma tabela ou gráfico, se aplicável].

    Este desfecho reflete a potência do método iterativo em resolver sistemas lineares de forma eficiente, oferecendo uma visão clara 
    das interdependências entre as variáveis e sua solução harmoniosa.
    """)