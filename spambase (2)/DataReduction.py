import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregar os dados
dados = pd.read_csv('C:\\Users\\CASA\Documents\\spambase\\spambase.data', header=None)

# Lidar com valores ausentes (remover linhas com valores ausentes)
dados.dropna(inplace=True)

# Separar características e alvo
X = dados.iloc[:, :-1]  # Características
y = dados.iloc[:, -1]   # Alvo

# Padronizar as características
padronizador = StandardScaler()
X_padronizado = padronizador.fit_transform(X)

# Aplicar PCA para compressão de atributos (supondo 10 componentes principais)
pca = PCA(n_components=10)
X_comprimido = pca.fit_transform(X_padronizado)

# Mostrar as dimensões antes e depois da compressão
print(f'Dimensões antes da compressão: {X.shape}')
print(f'Dimensões após a compressão: {X_comprimido.shape}')

# Supondo que você deseje manter 80% dos dados aleatoriamente
porcentagem_para_manter = 0.8
subconjunto_aleatorio = dados.sample(frac=porcentagem_para_manter, random_state=42)
