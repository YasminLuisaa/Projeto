import pandas as pd

# Substitua 'seu_arquivo.csv' pelo caminho real do seu arquivo de dados
df = pd.read_csv('C:\\Users\\CASA\\Documents\\spambase\\spambase.data')

# Exibir as primeiras linhas do DataFrame para verificar as colunas
print(df.head())

# Listar todas as colunas presentes no DataFrame
print(df.columns)
