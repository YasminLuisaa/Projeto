import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregue seu conjunto de dados, substitua 'seu_dataset.csv' pelo caminho ou nome real do seu arquivo
random_subset = pd.read_csv('C:\\Users\\CASA\Documents\\spambase\\spambase.data')

# Supondo que 'random_subset' seja o conjunto de dados após a redução
X_subset = random_subset.iloc[:, :-1]
scaler = StandardScaler()
X_standardized_subset = scaler.fit_transform(X_subset)
