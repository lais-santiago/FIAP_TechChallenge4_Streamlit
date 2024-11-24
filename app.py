import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import numpy as np

#Carregando os dados
@st.cache_data
def load_data():
    data = pd.read_csv('df_limpo.csv')
    data['data'] = pd.to_datetime(data['data'])
    data.set_index('data', inplace=True)
    return data

df = load_data()

st.write(df.info())

#Título do Dashboard
st.title('Dashboard Interativo: Preço do Petróleo Brent')

# Sidebar
st.sidebar.header("Filtros")
start_date = st.sidebar.date_input("Data Inicial", df.index.min().date())
end_date = st.sidebar.date_input("Data Final", df.index.max().date())
pred_date = st.sidebar.date_input(
    "Escolha uma data para previsão",
    min_value=df.index.min().date(),
    max_value=df.index.max().date()
)

#Filtro de data
filtered_data = df.loc[start_date:end_date]

#Gráfico de linha interativo usando o plotly
fig = px.line(filtered_data, 
                x=filtered_data.index, 
                y=filtered_data['preco'], 
                title="Variação do preço do petróleo Brent ao longo do tempo")
fig.update_layout(
    xaxis_title="Anos",
    yaxis_title="Preço do barril em dólares"
)
st.plotly_chart(fig)

st.subheader("Big Numbers")

#Valor médio do barril
preco_medio=filtered_data['preco'].mean()
st.metric(label="Valor médio do preço do barril de petróleo Brent", value=f"US$ {preco_medio:.2f}")

#Valor mínimo do barril
preco_minimo=filtered_data['preco'].min()
st.metric(label="Valor mínimo do preço do barril de petróleo Brent", value=f"US$ {preco_minimo:.2f}")

#Valor máximo do barril
preco_maximo=filtered_data['preco'].max()
st.metric(label="Valor máximo do preço do barril de petróleo Brent", value=f"US$ {preco_maximo:.2f}")

#Volatilidade diária do preço
df_volatil=pd.DataFrame()
df_volatil['retorno_log'] = np.log(df['preco']/df['preco'].shift(1))
df_volatil.dropna(inplace=True)
volatilidade_diaria=df_volatil['retorno_log'].std()
st.metric(label="Volatilidade diária do preço", value=f"US$ {volatilidade_diaria:.2f}")

#Volatilidade anual do preço
volatilidade_anual = volatilidade_diaria * np.sqrt(252)
st.metric(label="Volatilidade anual do preço", value=f"US$ {volatilidade_anual:.2f}")

#Variação percentual
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
valor_start_date=filtered_data.loc[start_date, 'preco']
valor_end_date=filtered_data.loc[end_date, 'preco']
variacao_percentual=((valor_end_date - valor_start_date)/valor_start_date)*100
st.metric(label="Variação percentual do preço", value=f"{variacao_percentual:.2f}%")


#Gráfico de histograma usando o plotly
fig2 = px.histogram(filtered_data,
                    x='preco',
                    nbins=30,
                    title="Frequência de preços do barril de petróleo Brent no período filtrado")
fig2.update_layout(
    xaxis_title="Preço em dólar do barril",
    yaxis_title="Frequência",
    template="plotly_white"
)
st.plotly_chart(fig2)

#Gráfico de barras usando o plotly
fig3 = px.bar(filtered_data,
                x='ano',
                y='preco',
                title="Preço médio anual do barril de petróleo Brent")
fig3.update_layout(
    xaxis_title="Ano",
    yaxis_title="Preço médio do barril em US\$",
    template="plotly_white"
)
st.plotly_chart(fig3)

#Carregar modelo
@st.cache_resource
def load_model():
    model = joblib.load("modelo/linear_regression.joblib")
    return model

modelo = load_model()

st.subheader("Resultado da Previsão")
if st.sidebar.button("Prever"):
    if pd.Timestamp(pred_date) in df.index.values:
        #Selecionar a linha correspondente a data
        linha = df.loc[df.index == pd.Timestamp(pred_date)]

        #Verificar se há valores válidos para lags e média móvel
        if linha[['preco_lag_1', 'preco_lag_2', 'preco_lag_3', 'preco_lag_4', 'preco_lag_5', 'media_movel_7d', 'dia_da_semana']].isnull().values.any():
            st.error("Dados históricos insuficientes para realizar uma previsão. Tente outra data.")
        else:
            #Criar vetor de entrada para o modelo
            input_data = linha[['preco_lag_1', 'preco_lag_2', 'preco_lag_3', 'preco_lag_4', 'preco_lag_5', 'media_movel_7d', 'dia_da_semana']]

            #Fazer a previsão
            previsao = modelo.predict(input_data)[0]

            #Mostrar resultado
            st.write(f"O preço previsto para {pred_date} é: US\$ **${previsao:.2f}**")
    else:
        st.error("A data selecionada não está nos dados históricos. Por favor, escolha outra data.")
else:
    st.write("Escolha uma data no filtro lateral para realizar a previsão e clique no botão prever")

# Insights
st.subheader("Insights Relevantes")
st.markdown("""
**Entre 2008 e 2009**

> Houve um pico no preço ficando acima de US\$ 140, seguido de uma queda para menos de US\$ 40 o barril. </br >
> Nesse período houve uma crise mundial do petróleo, chegando a afetar a produção de plásticos e produtos químicos. </br >
> A queda no preço do petróleo foi causada por um excesso de oferta, resultado da relutância dos membros da OPEP em reduzir a produção. A OPEP é um cartel formado pelos maiores produtores globais de petróleo, que se reúne para decidir sobre aumentos ou cortes de oferta. </br >
> A transição global da produção de energia proveniente de petróleo para a produção de energias alternativas também reduziu a demanda por petróleo.

**Entre 2014 e 2015**

> Houve uma nova queda no valor do barril de US\$ 120 até pouco menos de US\$ 50. </br >
> O preço do petróleo caiu devido a um aumento na produção, principalmente nos EUA, e a uma demanda menor do que o esperado na Europa e na Ásia. 

**De 2020 a 2021**

> O valor do barril sofreu a maior queda, chegando  a menos de US\$ 20, graças a pandemia do vírus COVID. </br >
> A pandemia de coronavírus causou uma queda de 21,5% na cotação do petróleo Brent em 2020. </br >
> O preço do petróleo subiu 41% em 2021, atingindo um recorde em dois anos. O WTI saiu de US\$ 48 por barril no início do ano para perto de US\$ 80 no final. O Brent, outro tipo de petróleo, subiu 57% em 2021. </br >
> A alta do preço do petróleo impactou a economia global e o Brasil, causando: 
> * Aumento da inflação 
> * Manutenção dos juros altos por mais tempo 
> * Impacto nos investimentos de todos os tipos 
> * Aumento do custo de vida das famílias 
> * Aumento do custo de produção das indústrias, comércio e serviços 
> * Desvalorização do real em comparação ao dólar </br >

> O preço do petróleo continuou a subir em 2022, tornando-se um fator de pressão inflacionária no Brasil e no mundo. 


##### Fontes
* REBOSSIO, Alejandro. O petróleo faz a América Latina ter seu pior ano desde 2009. El País, 07 de Janeiro de 2015. Disponível em: [link](https://brasil.elpais.com/brasil/2015/01/06/actualidad/1420570796_254696.html#:~:text=A%20regi%C3%A3o%20saiu%20rapidamente%20da%20crise%20mundial,consultorias%20e%20bancos%20feita%20pela%20empresa%20FocusEconomics.). Acesso em 24 nov. 2024.

* BOLDRINI, Gustavo. Como o preço do petróleo afeta o dólar (e vice-versa). InvestTalk, 30 de Setembro de 2024. Disponível em: [link](https://investalk.bb.com.br/noticias/quero-aprender/como-o-preco-do-petroleo-afeta-o-dolar-e-vice-versa#:~:text=Um%20dos%20principais%20catalisadores%20dos,de%20acordo%20com%20seus%20interesses.). Acesso em 24 nov. 2024.

* Redação G1. Entenda a queda do preço do petróleo e seus efeitos. G1, Economia, 16 de Janeiro de 2015. Disponível em: [link](https://g1.globo.com/economia/noticia/2015/01/entenda-queda-do-preco-do-petroleo-e-seus-efeitos.html#:~:text=Os%20principais%20apontados%20como%20%22culpados,na%20Europa%20e%20na%20%C3%81sia.). Acesso em 24 nov. 2024.

* Poder 360. Preço do petróleo sobe 41\% em 2021 e atinge recorde em 2 anos. udop - União Nacional da Bioenergia, 21 de Junho de 2021. Disponível em: [link](https://www.udop.com.br/noticia/2021/06/21/preco-do-petroleo-sobe-41-em-2021-e-atinge-recorde-em-2-anos.html). Acesso em 24 nov. 2024.

* ELIAS, Juliana. Preço do petróleo já subiu 60\% em 2021 - e há quem aposte em mais aumentos. CNN Brasil, 17 de Novembro de 2021. Disponível em: [link](https://www.cnnbrasil.com.br/economia/macroeconomia/preco-do-petroleo-ja-subiu-60-em-2021-e-ha-quem-aposte-em-mais-aumentos/#:~:text=S%C3%B3%20em%202021%2C%20sua%20alta,%2C%20da%20gasolina%2C%2073%25.). Acesso em 24 nov. 2024.

* MALAR, João Pedro. Tensão na Ucrânia pressiona petróleo, mas causas para alta já existiam; entenda. CNN Brasil, 28 de Janeiro de 2022. Disponível em: [link](https://www.cnnbrasil.com.br/economia/mercado/tensao-na-ucrania-pressiona-petroleo-mas-causas-para-alta-ja-existiam-entenda/#:~:text=Macroeconomia-,Tens%C3%A3o%20na%20Ucr%C3%A2nia%20pressiona%20petr%C3%B3leo%2C%20mas,para%20alta%20j%C3%A1%20existiam;%20entenda&text=O%20petr%C3%B3leo%20engatou%20em%202021,2022%20pr%C3%B3ximo%20dos%20US$%20100.). Acesso em 24 nov. 2024.
""")
