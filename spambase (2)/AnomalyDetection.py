import pandas as pd

# Ler o arquivo CSV em um DataFrame
df = pd.read_csv('C:\\Users\\CASA\\Documents\\spambase\\spambase.data')

# Verificar valores ausentes
valores_ausentes = df.isnull().sum()

# Exibir os valores ausentes
print(valores_ausentes)
