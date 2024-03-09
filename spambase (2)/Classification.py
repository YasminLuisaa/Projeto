import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Carregar os dados do arquivo .data
# Substitua 'seu_arquivo.data' pelo caminho real do seu arquivo de dados
df = pd.read_csv('C:\\Users\\CASA\Documents\\spambase\\spambase.data', header=None)

# Supondo que a última coluna seja a variável alvo 'spam'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar um modelo de Regressão Logística
modelo = LogisticRegression()
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_predito = modelo.predict(X_teste)

# Avaliar o modelo
print(f'Acurácia: {accuracy_score(y_teste, y_predito)}')
print(classification_report(y_teste, y_predito))
