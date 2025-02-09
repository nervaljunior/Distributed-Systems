import json
import time
import  requests
import pandas as pd

#5 intercity 

#Nesta etapa, os dados rotulados são enviados à plataforma [InterSCity](https://interscity.org/), através de API disponível.

""" urls:

https://cidadesinteligentes.lsdi.ufma.br/interscity/catalog

http://192.168.10.13/tagger/

https://cidadesinteligentes.lsdi.ufma.br/interscity/collector

https://cidadesinteligentes.lsdi.ufma.br/interscity/discovery

https://cidadesinteligentes.lsdi.ufma.br/interscity/actuator

https://cidadesinteligentes.lsdi.ufma.br//interscity/adaptor/subscriptions """

#5.1 Setup

#configurações iniciais Para o uso da API do interscity

# Endereço para a api
url = "https://cidadesinteligentes.lsdi.ufma.br"
api = "https://cidadesinteligentes.lsdi.ufma.br/interscity"

# Configurações da API
BASE_URL = "https://cidadesinteligentes.lsdi.ufma.br/interscity"
HEADERS = {"Content-Type": "application/json"}


def carregar_dados(file):
    
    df = pd.read_excel(file, skiprows=4)
    # Excluir a primeira linha, que parece ser um cabeçalho residual
    df = df.drop(index=0).drop(columns=df.columns[1])
    colunas_para_remover = ['feed', 'grid_import.3', 'grid_import.4']
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])


    # Definindo uma data inicial e criando um índice de 15 minutos para todo o DataFrame
    data_inicial = '2014-12-11 17:45'  # Inclui data e hora
    #2014-12-11T17:45:00Z
    #2014-12-11 (quinta-feira) dezembro
    frequencia = '15T'  # 15 minutos

    df.index = pd.date_range(start=data_inicial, periods=len(df), freq=frequencia)



    # Remover colunas desnecessárias (todas com NaN ou sem dados)
    df = df.dropna(axis=1, how='all')

    # Remover linhas que possuem somente valores NaN
    df = df.dropna(axis=0, how='all')

    # Remover colunas desnecessárias e colunas do tipo 'Unnamed'
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Dicionário de mapeamento para os novos nomes de colunas
    renomear_colunas = {
        # Industrial 1
        'grid_import': 'industrial1_grid_import',
        'pv_1': 'industrial1_pv_1',
        'pv_2': 'industrial1_pv_2',

        # Industrial 2
        'grid_import.1': 'industrial2_grid_import',
        'pv': 'industrial2_pv',
        'storage_charge': 'industrial2_storage_charge',
        'storage_decharge': 'industrial2_storage_decharge',

        # Industrial 3
        'area_offices': 'industrial3_area_offices',
        'area_room_1': 'industrial3_area_room_1',
        'area_room_2': 'industrial3_area_room_2',
        'area_room_3': 'industrial3_area_room_3',
        'area_room_4': 'industrial3_area_room_4',
        'compressor': 'industrial3_compressor',
        'cooling_aggregate': 'industrial3_cooling_aggregate',
        'cooling_pumps': 'industrial3_cooling_pumps',
        'dishwasher': 'industrial3_dishwasher',
        'ev': 'industrial3_ev',
        'grid_import.2': 'industrial3_grid_import',
        'machine_1': 'industrial3_machine_1',
        'machine_2': 'industrial3_machine_2',
        'machine_3': 'industrial3_machine_3',
        'machine_4': 'industrial3_machine_4',
        'machine_5': 'industrial3_machine_5',
        'pv_facade': 'industrial3_pv_facade',
        'pv_roof': 'industrial3_pv_roof',
        'refrigerator': 'industrial3_refrigerator',
        'ventilation': 'industrial3_ventilation',

        # Public 1 e 2
        'grid_import.3': 'public1_grid_import',
        'grid_import.4': 'public2_grid_import',

        # Residential 1
        'dishwasher.1': 'residential1_dishwasher',
        'freezer': 'residential1_freezer',
        'grid_import.5': 'residential1_grid_import',
        'heat_pump': 'residential1_heat_pump',
        'pv.1': 'residential1_pv',
        'washing_machine': 'residential1_washing_machine',

        # Residential 2
        'circulation_pump': 'residential2_circulation_pump',
        'dishwasher.2': 'residential2_dishwasher',
        'freezer.1': 'residential2_freezer',
        'grid_import.6': 'residential2_grid_import',
        'washing_machine.1': 'residential2_washing_machine',

        # Residential 3
        'circulation_pump.1': 'residential3_circulation_pump',
        'dishwasher.3': 'residential3_dishwasher',
        'freezer.2': 'residential3_freezer',
        'grid_export': 'residential3_grid_export',
        'grid_import.7': 'residential3_grid_import',
        'pv.2': 'residential3_pv',
        'refrigerator.1': 'residential3_refrigerator',
        'washing_machine.2': 'residential3_washing_machine',

        # Residential 4
        'dishwasher.4': 'residential4_dishwasher',
        'ev.1': 'residential4_ev',
        'freezer.3': 'residential4_freezer',
        'grid_export.1': 'residential4_grid_export',
        'grid_import.8': 'residential4_grid_import',
        'heat_pump.1': 'residential4_heat_pump',
        'pv.3': 'residential4_pv',
        'refrigerator.2': 'residential4_refrigerator',
        'washing_machine.3': 'residential4_washing_machine',

        # Residential 5
        'dishwasher.5': 'residential5_dishwasher',
        'grid_import.9': 'residential5_grid_import',
        'refrigerator.3': 'residential5_refrigerator',
        'washing_machine.4': 'residential5_washing_machine',

        # Residential 6
        'circulation_pump.2': 'residential6_circulation_pump',
        'dishwasher.6': 'residential6_dishwasher',
        'freezer.4': 'residential6_freezer',
        'grid_export.2': 'residential6_grid_export',
        'grid_import.10': 'residential6_grid_import',
        'pv.4': 'residential6_pv',
        'washing_machine.5': 'residential6_washing_machine'
    }

    # Renomear as colunas no DataFrame
    df.rename(columns=renomear_colunas, inplace=True)

    df_energia = df.diff()

    # Remover a primeira linha gerada pela diferenciação
    df_energia.dropna(inplace=True)

    return df_energia


# Etapa 1: Criar capacidade de energia
def criar_capacidade():
    capacidade_data = {
        "name": "Energia(kwh)",
        "description": "Medição de energia em kWh",
        "capability_type": "sensor"
    }
    try:
        response = requests.post(f"{BASE_URL}/catalog/capabilities", json=capacidade_data, headers=HEADERS)
        response.raise_for_status()
        if response.status_code == 201:
            print("Capacidade criada com sucesso.")
        else:
            print("Erro ao criar capacidade:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("Erro de requisição na criação de capacidade:", e)

# Etapa 2: Registrar recurso e obter UUIDs
def registrar_recurso(df):
    uuids = []
    for column in df.columns:
        resource_data = {
            "data": {
                "description": [f"medidor de energia para {column}"],
                "capabilities": ["Energia(kwh)"],
                "status": "active",
                "city": "São Luis",
                "country": "Brazil",
                "lat": -23.5,
                "lon": -46.6,
                "collect_interval": 15,
                "url": "energia.com/medidorEnergia",
                "uri": "energia.com/medidorEnergia"
            }
        }
        try:
            response = requests.post(f"{BASE_URL}/catalog/resources", json=resource_data, headers=HEADERS)
            response.raise_for_status()
            resource = response.json()
            uuid = resource['data']['uuid']
            print("Recurso criado com UUID:", uuid)
            uuids.append(uuid)
        except requests.exceptions.RequestException as e:
            print(f"Erro de requisição ao registrar recurso para '{column}':", e)
        except json.JSONDecodeError:
            print("Erro ao decodificar JSON ao registrar recurso para", column)
        except KeyError:
            print("UUID não encontrado na resposta da API para o recurso", column)
    return uuids

# Etapa 3: Enviar dados simulados de consumo de energia em paralelo
def enviar_dados(df, uuids):
    num_rows = len(df)
    for index in range(num_rows):  # Itera sobre cada linha do DataFrame
        for column, uuid in zip(df.columns, uuids):  # Para cada UUID e coluna
            try:
                energia_value = df.iloc[index][column]  # Ajustado para iloc
                print(energia_value)
                if pd.isna(energia_value):
                    print(f"Valor NaN ignorado para UUID {uuid} na linha {index+1}")
                    continue

                capability_data = {
                    "data": [
                        {
                            "Energia": float(energia_value),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "date": df.index[index].strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                    ]
                }
                response = requests.post(
                    f"{BASE_URL}/adaptor/resources/{uuid}/data/environment_monitoring",
                    json=capability_data,
                    headers=HEADERS
                )
                response.raise_for_status()
                if response.status_code == 201:
                    print(f"Dado {index+1}/{num_rows} enviado para UUID {uuid} com sucesso.")
                else:
                    print(f"Erro ao enviar dados para UUID {uuid}: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Erro ao processar dado para UUID {uuid}: {e}")

        time.sleep(5)



if __name__ == "__main__":
    
    path='./data/household_data.xlsx'
    
    df= carregar_dados(path)
    criar_capacidade()
    uuids = registrar_recurso(df)
    enviar_dados(df, uuids)
    
