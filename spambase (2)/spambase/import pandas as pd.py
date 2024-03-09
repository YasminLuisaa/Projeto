import pandas as pd

# Carregue o conjunto de dados
nomes_colunas = [
    f'word_freq_{i}' for i in range(48)
] + [
    f'char_freq_{chr(i)}' for i in range(ord('A'), ord('Z') + 1)
] + [
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total',
    'spam'
]

df = pd.read_csv(r'C:\Users\CASA\Documents\spambase\spambase.data', header=None, names=nomes_colunas)

# Exiba informações básicas sobre o conjunto de dados
print(df.info())
print(df.describe())
