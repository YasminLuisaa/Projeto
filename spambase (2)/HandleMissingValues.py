# Import the pandas library
import pandas as pd

# Load the data
colunas = [f'word_freq_{i}' for i in range(1, 49)] + \
           [f'char_freq_{chr(i)}' for i in range(ord('a'), ord('z')+1)] + \
           ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']

df = pd.read_csv(r'C:\Users\CASA\Documents\spambase\spambase.data', header=None, names=colunas)

# Replace '?' with NaN and drop missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
