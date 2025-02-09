import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configuração da API do InterSCity
BASE_URL = "https://cidadesinteligentes.lsdi.ufma.br/interscity"
HEADERS = {"Content-Type": "application/json"}


@st.cache_data
def carregar_dados():
    """Busca dados da API do InterSCity usando UUIDs."""
    try:
        response = requests.get(f"{BASE_URL}/catalog/resources", headers=HEADERS)
        response.raise_for_status()
        recursos = response.json().get("resources", [])

        if not recursos:
            st.error("Nenhum recurso encontrado na API do InterSCity.")
            return None

        dados_coletados = []

        for recurso in recursos:
            uuid = recurso.get("uuid")
            descricao = recurso.get("data", {}).get("description", ["Sem descrição"])[0]

            if not uuid:
                continue

            response = requests.get(f"{BASE_URL}/collector/resources/{uuid}/data", headers=HEADERS)

            if response.status_code != 200:
                continue  # Pula se a resposta não for bem-sucedida

            dados = response.json().get("data", [])

            for d in dados:
                timestamp = d.get("timestamp")
                energia = d.get("Energia", None)
                if energia is not None:
                    dados_coletados.append(
                        {"uuid": uuid, "descricao": descricao, "timestamp": timestamp, "Energia": energia})

        if not dados_coletados:
            st.error("Nenhum dado de medição encontrado.")
            return None

        df = pd.DataFrame(dados_coletados)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao conectar com a API: {e}")
        return None
    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
        return None


# Classe para modelar o sistema de energia
class EnergyModel:
    def __init__(self, pv=0, importacao=0, exportacao=0):
        self.pv = pv
        self.importacao = importacao
        self.exportacao = exportacao

    def calcular_cargas(self):
        return self.pv + self.importacao - self.exportacao

    def balanco_energetico(self):
        return {
            "PV": self.pv,
            "Importação": self.importacao,
            "Exportação": self.exportacao,
            "Cargas": self.calcular_cargas()
        }


# Processamento de dados
def processar_dados(df):
    if df is None or df.empty:
        return None

    df["Cargas Calculadas"] = df.get("Energia", 0)

    return df


# Previsão de consumo com LSTM
def treinar_modelo_lstm(dados):
    if dados is None or dados.empty:
        st.error("Erro: Dados insuficientes para o treinamento do modelo.")
        return None, None

    scaler = MinMaxScaler()
    dados_scaled = scaler.fit_transform(dados.values.reshape(-1, 1))
    X, y = [], []

    for i in range(7, len(dados_scaled)):
        X.append(dados_scaled[i - 7:i])
        y.append(dados_scaled[i])

    X, y = np.array(X), np.array(y)

    modelo = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(X, y, epochs=20, verbose=0)

    return modelo, scaler


def prever_consumo(modelo, dados, scaler):
    if modelo is None or dados.empty:
        return None

    dados_scaled = scaler.transform(dados[-7:].values.reshape(-1, 1))
    entrada = dados_scaled.reshape(1, 7, 1)
    previsao_scaled = modelo.predict(entrada, verbose=0)
    return scaler.inverse_transform(previsao_scaled)[0, 0]


# Configuração do Streamlit
st.title("Análise de Consumo e Geração de Energia Residencial")

# Carregar dados da API do InterSCity
df = carregar_dados()

if df is not None and not df.empty:
    df['DataHora'] = df.index

    # Sidebar para seleção de residência e intervalo
    residencias = df["descricao"].unique()
    residencia_selecionada = st.sidebar.selectbox("Selecione o Recurso", residencias)

    data_inicio = st.sidebar.date_input("Data de Início", df.index.min().date())
    data_fim = st.sidebar.date_input("Data de Fim", df.index.max().date())

    # Garantir que data_inicio e data_fim sejam do tipo datetime
    data_inicio = pd.to_datetime(data_inicio)
    data_fim = pd.to_datetime(data_fim)

    # Filtrar dados com base na seleção
    df_residencia = df[(df["descricao"] == residencia_selecionada) & (df.index >= data_inicio) & (df.index <= data_fim)]

    if not df_residencia.empty:
        df_residencia = processar_dados(df_residencia)

        # Exibir balanço energético
        st.subheader("Balanço Energético")
        energia_total = df_residencia["Energia"].sum()

        st.write(f"Energia total consumida: {energia_total:.2f} kWh")

        # Treinar modelo LSTM e exibir gráficos
        if st.button("Treinar Modelo e Exibir Gráficos"):
            modelo, scaler = treinar_modelo_lstm(df_residencia["Energia"])
            if modelo is not None:
                fig, ax = plt.subplots()
                x = np.arange(len(df_residencia))
                y_real = df_residencia["Energia"].values
                y_previsto = [prever_consumo(modelo, df_residencia["Energia"][:i + 1], scaler) for i in
                              range(7, len(df_residencia))]

                ax.plot(x, y_real, label="Consumo Real", color='red')
                ax.plot(x[7:], y_previsto, label="Previsão", color='blue', linestyle='dashed')
                ax.set_xlabel("Tempo")
                ax.set_ylabel("Energia (kWh)")
                ax.legend()
                st.pyplot(fig)
    else:
        st.error("Nenhum dado disponível para o intervalo selecionado.")
