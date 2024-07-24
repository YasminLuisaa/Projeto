import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Carregar dados
input_file = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'
column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
    'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total', 'spam'
]
data = pd.read_csv(input_file, names=column_names)

# Verificar as colunas do DataFrame
print("Colunas do DataFrame:", data.columns)

# Verificar as primeiras linhas do DataFrame
print("Primeiras linhas do DataFrame:")
print(data.head())

# Passo 2: Separar características e rótulos
# Ajustar 'spam' para o nome correto da coluna, se necessário
label_column = 'spam'  # Ajuste conforme necessário

X = data.drop(label_column, axis=1)
y = data[label_column]

# Passo 3: Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Passo 4: Reduzir a dimensionalidade
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Passo 5: Método do cotovelo para encontrar o número ótimo de clusters
distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para encontrar o número ótimo de clusters')
plt.show()

# Passo 6: Avaliar o coeficiente de silhueta
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Score de Silhueta')
plt.title('Método da Silhueta para encontrar o número ótimo de clusters')
plt.show()

# Melhor valor de k
best_k = np.argmax(silhouette_scores) + 2
print(f'Melhor número de clusters: {best_k}')
print(f'Melhor score de silhueta: {max(silhouette_scores)}')
