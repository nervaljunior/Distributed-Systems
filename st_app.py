import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Função para carregar dados
@st.cache_data
def carregar_dados(caminho):
    return pd.read_csv(caminho, index_col=0, parse_dates=True)

# Classe para modelar o sistema de energia
class EnergyModel:
    def __init__(self, pv=None, importacao=None, exportacao=None):
        self.pv = pv if pv is not None else 0
        self.importacao = importacao if importacao is not None else 0
        self.exportacao = exportacao if exportacao is not None else 0

    def calcular_cargas(self):
        # Caso não haja PV, a carga será igual à importação
        if self.pv == 0:
            return self.importacao
        return self.pv + self.importacao - self.exportacao

    def balanco_energetico(self):
        cargas = self.calcular_cargas()
        return {
            "PV": self.pv,
            "Importação": self.importacao,
            "Exportação": self.exportacao,
            "Cargas": cargas
        }

# Função para classificar o dia
def classificar_dia(df):
    total_importacao = df.get("grid_import", pd.Series(0)).sum()
    total_exportacao = df.get("grid_export", pd.Series(0)).sum()
    total_pv = df.get("pv", pd.Series(0)).sum()
    total_cargas = df.get("Cargas Calculadas", pd.Series(0)).sum()


    if total_exportacao > total_cargas:
        return "🟢 Alta Exportação"
    elif total_importacao > total_pv:
        return "🔴 Alta Importação"
    else:
        return "🟡 Equilíbrio (Balanço Estável)"

# Processar dados
def processar_dados(df):
    resultados = []
    for _, row in df.iterrows():
        modelo = EnergyModel(
            pv=row.get('pv', 0),
            importacao=row.get('grid_import', 0),
            exportacao=row.get('grid_export', 0)
        )
        resultado = modelo.balanco_energetico()
        resultados.append(resultado)
    df['Cargas Calculadas'] = [r['Cargas'] for r in resultados]
    if 'Grid Balance' not in df.columns:
        df['Grid Balance'] = df['grid_import'] - df['grid_export']

    return df

# Treinamento do modelo LSTM
def treinar_modelo_lstm(dados):
    scaler = MinMaxScaler()
    dados_scaled = scaler.fit_transform(dados.values.reshape(-1, 1))
    X, y = [], []
    for i in range(7, len(dados_scaled)):
        X.append(dados_scaled[i-7:i])
        y.append(dados_scaled[i])
    X, y = np.array(X), np.array(y)

    modelo = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(X, y, epochs=20, verbose=0)
    return modelo, scaler


# Prever consumo
def prever_consumo(modelo, dados, scaler):
    dados_scaled = scaler.transform(dados[-7:].values.reshape(-1, 1))
    entrada = dados_scaled.reshape(1, 7, 1)
    previsao_scaled = modelo.predict(entrada, verbose=0)
    return scaler.inverse_transform(previsao_scaled)[0, 0]

# Layout de botões
col1, col2, col3 = st.columns(3)
if col1.button("▶️ Executar"):
    st.write("Ação: Executar")
if col2.button("⏸️ Pausar"):
    st.write("Ação: Pausar")
if col3.button("⏹️ Parar"):
    st.write("Ação: Parar")


# Abas de Navegação
tab = st.selectbox("Dia da Semana", ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"])


# Gráfico de Consumo e Previsão
def exibir_grafico(df, modelo, scaler):
    st.subheader("Consumo Real vs Previsão")
    fig, ax = plt.subplots()
    x = np.arange(len(df))
    y_real = df['Cargas Calculadas'].values
    y_previsto = [prever_consumo(modelo, df['Cargas Calculadas'][:i+1], scaler) for i in range(7, len(df))]

    ax.plot(x, y_real, label="Consumo Real", color='red')
    ax.plot(x[7:], y_previsto, label="Previsão", color='blue', linestyle='dashed')
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Consumo (kW)")
    ax.legend()
    st.pyplot(fig)

# Plotar dados com verificações
def plotar_dados(df, modelo, scaler):
    fig, ax = plt.subplots(figsize=(12, 6))

    if 'grid_import' in df.columns:
        ax.plot(df.index, df["grid_import"], label="Importação", color="blue")
    if 'grid_export' in df.columns:
        ax.plot(df.index, df["grid_export"], label="Exportação", color="orange")
    if 'pv' in df.columns:
        ax.plot(df.index, df["pv"], label="PV", color="green")
    if 'Cargas Calculadas' in df.columns:
        ax.plot(df.index, df["Cargas Calculadas"], label="Cargas Calculadas", color="red")
    if 'Grid Balance' in df.columns:
        ax.plot(df.index, df["Grid Balance"], label="Balanço da Rede", linestyle="--", color="purple")

    if modelo:
        previsoes = []
        for i in range(7, len(df)):
            previsoes.append(prever_consumo(modelo, df["Cargas Calculadas"][:i], scaler))
        ax.plot(df.index[7:], previsoes, label="Previsão LSTM", color="cyan", linestyle="dashed")

    ax.set_title("Consumo e Geração de Energia")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Energia (kW)")
    ax.legend()
    st.pyplot(fig)


# Configuração do Streamlit
st.title("Análise de Consumo e Geração de Energia Residencial")

# Carregar dados
caminho_dados = './data/potencia_residential_data.csv'
df = carregar_dados(caminho_dados)

# Sidebar para seleção de residência e intervalo
df['DataHora'] = df.index
residencias = [col.split("_")[0] for col in df.columns if "residential" in col]
residencia_selecionada = st.sidebar.selectbox("Selecione a Residência", list(set(residencias)))
data_inicio = st.sidebar.date_input("Data de Início", df.index.min().date())
data_fim = st.sidebar.date_input("Data de Fim", df.index.max().date())

# Filtrar dados com base na seleção
df_residencia = df[[col for col in df.columns if col.startswith(residencia_selecionada)]].copy()
df_residencia.columns = [col.replace(f"{residencia_selecionada}_", "") for col in df_residencia.columns]
df_residencia = df_residencia.loc[data_inicio:data_fim]

# Processar os dados
try:
    df_residencia = processar_dados(df_residencia)

    # Exibir balanço energético
    st.subheader("Balanço Energético")
    pv = df_residencia.get('pv', pd.Series(0)).sum()  # Usar 0 se 'pv' não existir
    importacao = df_residencia.get('grid_import', pd.Series(0)).sum()
    exportacao = df_residencia.get('grid_export', pd.Series(0)).sum()
    modelo_energia = EnergyModel(pv=pv, importacao=importacao, exportacao=exportacao)
    balanco = modelo_energia.balanco_energetico()

    for key, value in balanco.items():
        st.write(f"{key}: {value}")

    # Classificar dia
    st.subheader("Classificação do Dia")
    classificacao = classificar_dia(df_residencia)
    st.write(f"Classificação: {classificacao}")

    # Gráfico de barras para categorias
    st.subheader("Gráfico de Consumo e Geração por Categoria")
    fig, ax = plt.subplots()
    df_residencia.sum().plot(kind="bar", ax=ax, color=["blue", "orange", "green", "red"])
    ax.set_title("Consumo e Geração")
    ax.set_ylabel("Energia (kW)")
    ax.set_xlabel("Categorias")
    st.pyplot(fig)

    # Gráficos de linha para importação, exportação, cargas e PV
    st.subheader("Gráficos de Linha: Importação, Exportação, Cargas e PV")
    fig, ax = plt.subplots()

    # Verificar se a coluna 'pv' existe
    if 'pv' in df_residencia.columns:
        df_residencia[['grid_import', 'grid_export', 'Cargas Calculadas', 'pv']].plot(ax=ax)
        ax.legend(['Importação', 'Exportação', 'Cargas Calculadas', 'Geração PV'])
    else:
        df_residencia[['grid_import', 'grid_export', 'Cargas Calculadas']].plot(ax=ax)
        ax.legend(['Importação', 'Exportação', 'Cargas Calculadas'])

    ax.set_title("Importação, Exportação, Cargas e Geração PV ao Longo do Tempo")
    ax.set_ylabel("Energia (kWh)")
    ax.set_xlabel("Tempo")
    st.pyplot(fig)
    
    

    
        # Treinar modelo LSTM e exibir gráficos
    if st.button("Treinar Modelo e Exibir Gráficos"):
        modelo, scaler = treinar_modelo_lstm(df_residencia["Cargas Calculadas"])
        plotar_dados(df_residencia, modelo, scaler)
    else:
        plotar_dados(df_residencia, None, None)

except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
