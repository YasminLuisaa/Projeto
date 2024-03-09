import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregue seu conjunto de dados
# Substitua 'seu_dataset.csv' pelo nome real do seu arquivo de dados
df = pd.read_csv('C:\\Users\\CASA\\Documents\\spambase\\spambase.data')

# Verifique os nomes das colunas
print(df.columns)

# Certifique-se de que as colunas 'capital_run_length_longest' e 'spam' existem no conjunto de dados
if 'capital_run_length_longest' in df.columns and 'spam' in df.columns:
    # Use apenas as características relevantes para a regressão
    X_regressao = df.drop(['capital_run_length_longest', 'spam'], axis=1)
    y_regressao = df['capital_run_length_longest']

    # Divida os dados em conjuntos de treinamento e teste
    X_treino_reg, X_teste_reg, y_treino_reg, y_teste_reg = train_test_split(X_regressao, y_regressao, test_size=0.2, random_state=42)

    # Inicialize e treine um modelo de Regressão Linear
    modelo_regressao = LinearRegression()
    modelo_regressao.fit(X_treino_reg, y_treino_reg)

    # Faça previsões no conjunto de teste
    y_predito_reg = modelo_regressao.predict(X_teste_reg)

    # Avalie o modelo de regressão
    print(f'Erro Quadrático Médio: {mean_squared_error(y_teste_reg, y_predito_reg)}')
else:
    print("Certifique-se de que as colunas 'capital_run_length_longest' e 'spam' existem no conjunto de dados.")
