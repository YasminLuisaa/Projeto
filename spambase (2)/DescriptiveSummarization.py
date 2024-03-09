import pandas as pd

# Carregar o conjunto de dados
nomes_colunas = [f'word_freq_WORD_{i}' for i in range(48)] + \
                [f'char_freq_CHAR_{i}' for i in range(6)] + \
                ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']

df = pd.read_csv('C:\\Users\\CASA\Documents\\spambase\\spambase.data', names=nomes_colunas)

# Exibir informações básicas sobre o conjunto de dados
print(df.info())

# Exibir estatísticas sumárias
print(df.describe())
