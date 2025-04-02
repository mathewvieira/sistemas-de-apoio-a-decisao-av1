import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


st.set_page_config(page_title='Previsor de Desempenho no ENEM', page_icon='📚', layout='wide')


@st.cache_data
def carregar_dados(caminho_arquivo):
    '''Carrega e prepara os dados do ENEM.'''

    try:
        # Carregar apenas as colunas necessárias para economizar memória
        colunas_interesse = ['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO', 'Q006', 'Q002', 'TP_ESCOLA', 'TP_COR_RACA', 'SG_UF_PROVA', 'CO_MUNICIPIO_PROVA']

        df = pd.read_csv(caminho_arquivo, encoding='iso-8859-1', delimiter=';', usecols=colunas_interesse, low_memory=False)

        # Remover linhas com valores nulos nas notas
        df = df.dropna(subset=['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO'])
        return df

    except Exception as e:
        st.error(f'Erro ao carregar os dados: {e}')
        return None


def preparar_dados(df):
    '''Prepara os dados para treinamento do modelo.'''

    # Variáveis independentes (features)
    X = df[['Q006', 'Q002', 'TP_ESCOLA', 'TP_COR_RACA', 'SG_UF_PROVA', 'CO_MUNICIPIO_PROVA']]

    # Variáveis dependentes (targets)
    y = df[['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO']]

    # Converter variáveis categóricas para numéricas
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def criar_modelos(X, y, n_neighbors=5):
    '''Cria modelos KNN para cada área do conhecimento.'''

    modelos = {}

    for area in y.columns:
        # Pipeline com imputação de valores ausentes, normalização e KNN
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=n_neighbors))
        ])

        # Treinar o modelo
        pipeline.fit(X, y[area])
        modelos[area] = pipeline

    return modelos


def prever_notas(modelos, X_novo):
    '''Prevê as notas de um novo aluno.'''

    previsoes = {}

    for area, modelo in modelos.items():
        previsao = modelo.predict(X_novo)[0]
        previsoes[area] = round(previsao, 1)

    return previsoes


def obter_vizinhos(X, y, X_novo, n_neighbors=5):
    '''Obtém os vizinhos mais próximos para comparação.'''

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_novo_scaled = scaler.transform(X_novo)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)

    indices = nn.kneighbors(X_novo_scaled, return_distance=False)[0]

    return y.iloc[indices]


def interface_entrada():
    '''Interface para entrada de dados do aluno.'''

    st.title('Previsor de Desempenho no ENEM')
    st.markdown('### Insira os dados do aluno para prever o desempenho.')

    with st.form('dados_aluno'):
        col1, col2 = st.columns(2)

        with col1:
            renda = st.selectbox(index=6, label='Renda Familiar:', options=['A - Nenhuma renda', 'B - Até R$ 1.250,00', 'C - De R$ 01.250,01 até R$ 01.850,00', 'D - De R$ 01.850,01 até R$ 02.450,00', 'E - De R$ 02.450,01 até R$ 03.050,00', 'F - De R$ 03.050,01 até R$ 03.650,00', 'G - De R$ 03.650,01 até R$ 04.850,00', 'H - De R$ 04.850,01 até R$ 06.050,00', 'I - De R$ 06.050,01 até R$ 07.250,00', 'J - De R$ 07.250,01 até R$ 08.450,00', 'K - De R$ 08.450,01 até R$ 09.650,00', 'L - De R$ 09.650,01 até R$ 10.950,00', 'M - De R$ 10.950,01 até R$ 12.150,00', 'N - De R$ 12.150,01 até R$ 14.550,00', 'O - De R$ 14.550,01 até R$ 18.150,00', 'P - De R$ 18.150,01 até R$ 24.250,00', 'Q - Acima de R$ 24.250,01'])

            # Mapeamento da letra para o código numérico usado nos microdados
            renda_map = { letra: idx+1 for idx, letra in enumerate('ABCDEFGHIJKLMNOPQ')}
            renda_codigo = renda_map[renda[0]]

            escolaridade_mae = st.selectbox(index=6, label='Escolaridade da Mãe:', options=['A - Não estudou', 'B - Ensino Fundamental I incompleto', 'C - Ensino Fundamental I completo', 'D - Ensino Fundamental II incompleto', 'E - Ensino Fundamental II completo', 'F - Ensino Médio incompleto', 'G - Ensino Médio completo', 'H - Ensino Superior incompleto', 'I - Ensino Superior completo', 'J - Pós-graduação'])

            escolaridade_map = {letra: idx+1 for idx, letra in enumerate('ABCDEFGHIJ')}
            escolaridade_codigo = escolaridade_map[escolaridade_mae[0]]

            tipo_escola = st.radio(index=1, label='Tipo de Escola:', options=['Não respondeu', 'Pública', 'Privada'])
            tipo_escola_map = {'Não respondeu': 0, 'Pública': 1, 'Privada': 2}
            tipo_escola_codigo = tipo_escola_map[tipo_escola]

        with col2:
            raca_cor = st.selectbox(index=2, label='Raça/Cor:', options=['Não declarado', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indígena'])
            raca_map = {'Não declarado': 0, 'Branca': 1, 'Preta': 2, 'Parda': 3, 'Amarela': 4, 'Indígena': 5}
            raca_codigo = raca_map[raca_cor]

            estado = st.selectbox(index=25, label='Estado:', options=['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'])

            # Código Default 3550308 (São Paulo)
            municipio = st.text_input('Código do Município:', value='3550308')

            try:
                municipio_codigo = int(municipio)

            except Exception:
                municipio_codigo = 3550308

        k_neighbors = st.slider('Número de vizinhos (k):', min_value=1, max_value=20, value=5)

        submit_button = st.form_submit_button('Prever Desempenho')

        if submit_button:
            return (
                { 'Q006': renda_codigo, 'Q002': escolaridade_codigo, 'TP_ESCOLA': tipo_escola_codigo, 'TP_COR_RACA': raca_codigo, 'SG_UF_PROVA': estado, 'CO_MUNICIPIO_PROVA': municipio_codigo},
                k_neighbors
            )

    return None, None


def main():
    st.sidebar.text('Algoritmo K-Nearest Neighbors (K-NN) para previsão de notas de aluno.')
    st.sidebar.title('Configurações')

    arquivo_dados = st.sidebar.file_uploader(label='Carregar arquivo de microdados do ENEM (CSV)', type='csv')

    if arquivo_dados is None:
        st.warning('Por favor, carregue um arquivo CSV com os microdados do ENEM.')
        st.info('Após carregar o arquivo, aguarde alguns instantes!')
        st.info('Você pode baixar os microdados oficiais do ENEM no site do INEP:\n\nhttps://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem\n\n(Recomendado: MICRODADOS_ENEM_2023.csv)')
        st.info('Importante! Arquivo CSV deve conter os campos:\n\nNU_NOTA_MT, NU_NOTA_CN, NU_NOTA_LC, NU_NOTA_CH, NU_NOTA_REDACAO,\n\nQ006, Q002, TP_ESCOLA, TP_COR_RACA, SG_UF_PROVA, CO_MUNICIPIO_PROVA.')
        st.info('Dica! Caso seu arquivo seja maior que 200MB, execute o seguinte comando:\n\nstreamlit run main.py --server.maxUploadSize 2048')
        return

    with st.spinner('Carregando dados do ENEM...'):
        df = carregar_dados(arquivo_dados)

    if df is None or df.empty:
        st.error('Não foi possível carregar os dados ou o arquivo está vazio.')
        return

    st.sidebar.success(f'Dados carregados com sucesso! Total de registros: {len(df)}')

    # Limitar o tamanho do dataframe para melhorar a performance
    tamanho_amostra = st.sidebar.slider(min_value=1000, max_value=min(50000, len(df)), value=min(10000, len(df)), step=1000, label='Tamanho da amostra para treinamento:')

    df_amostra = df.sample(tamanho_amostra, random_state=42)

    dados_aluno, k = interface_entrada()

    if dados_aluno:
        with st.spinner('Preparando dados e treinando modelo...'):
            X, y = preparar_dados(df_amostra)

            # Preparar dados do novo aluno
            novo_aluno_df = pd.DataFrame([dados_aluno])
            novo_aluno_df = pd.get_dummies(novo_aluno_df, drop_first=True)

            # Alinhar colunas do novo aluno com o conjunto de treinamento
            for col in X.columns:
                if col not in novo_aluno_df.columns:
                    novo_aluno_df[col] = 0

            novo_aluno_df = novo_aluno_df[X.columns]

            # Criar e treinar modelos
            modelos = criar_modelos(X, y, n_neighbors=k)

            # Fazer previsões
            previsoes = prever_notas(modelos, novo_aluno_df)

            # Obter vizinhos para comparação
            vizinhos = obter_vizinhos(X, y, novo_aluno_df, n_neighbors=k)

        # Exibir resultados
        st.success('Previsão concluída!')

        st.subheader('Notas Previstas')

        areas = { 'NU_NOTA_MT': 'Matemática', 'NU_NOTA_CN': 'Ciências da Natu.', 'NU_NOTA_LC': 'Linguagens e Cód.', 'NU_NOTA_CH': 'Ciências Humanas', 'NU_NOTA_REDACAO': 'Redação'}

        # Média geral
        media_prevista = sum(previsoes.values()) / len(previsoes)
        st.metric('Média Geral Prevista', f'{media_prevista:.1f}')

        # Visualização das previsões
        st.subheader('Detalhamento por Área')
        colunas = st.columns(5)

        for i, (codigo, nome) in enumerate(areas.items()):
            nota_prevista = previsoes[codigo]
            media_vizinhos = vizinhos[codigo].mean()

            with colunas[i]:
                st.metric(label=nome, value=f'{nota_prevista:.1f}', help=f'Média dos {k} vizinhos mais próximos: {media_vizinhos:.1f}')

if __name__ == '__main__':
    main()
