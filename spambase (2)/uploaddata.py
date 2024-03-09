# Import the pandas library
import pandas as pd

# Substitua 'path_to_dataset' pelo caminho real do seu arquivo de conjunto de dados
data_path = 'C:\\Users\\CASA\\Documents\\spambase\\spambase.data'

column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
    'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
    'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
    'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu',
    'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
    'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total', 'spam'
]

# Carregue o conjunto de dados no DataFrame
spam_data = pd.read_csv(data_path, names=column_names, na_values='?')

# Continue with the rest of your code
colunas = [f'word_freq_{i}' for i in range(1, 49)] + \
           [f'char_freq_{chr(i)}' for i in range(ord('a'), ord('z')+1)] + \
           ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']

df = pd.read_csv(r'C:\Users\CASA\Documents\spambase\spambase.data', header=None, names=colunas[:-1])  # Remove the extra backslash

# Now you have both datasets loaded in 'spam_data' and 'df'
# Imprimir as primeiras 5 linhas do DataFrame 'spam_data'
print(spam_data.head())
# Imprimir estatísticas descritivas do DataFrame 'spam_data'
print(spam_data.describe())
