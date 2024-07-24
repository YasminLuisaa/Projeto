import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, homogeneity_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Carregar os dados
spambase_data = pd.read_csv(input_file, names=column_names)

# Visualização inicial
sns.scatterplot(data=spambase_data, x='word_freq_your', y='word_freq_000', hue='spam')
plt.title('SPAM vs Não-SPAM baseado em word_freq_your e word_freq_000')
plt.xlabel('word_freq_your')
plt.ylabel('word_freq_000')
plt.show()

# Análise de correlação para seleção de características
correlation_matrix = spambase_data.corr()
top_features = correlation_matrix['spam'].abs().sort_values(ascending=False).index[1:11]

# Visualizar a correlação das características selecionadas
plt.figure(figsize=(12, 8))
sns.heatmap(spambase_data[top_features].corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação das Principais Características')
plt.show()

# Dados reduzidos com as características mais relevantes
X_reduced = spambase_data[top_features]
y = spambase_data['spam']

# Normalização dos dados
scaler = StandardScaler()
X_reduced_scaled = scaler.fit_transform(X_reduced)

# Redução de dimensionalidade com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_reduced_scaled)

# Aplicação de KMeans variando o número de clusters
silhouette_scores = []
inertias = []
homogeneity_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_pca, labels))
    inertias.append(kmeans.inertia_)
    homogeneity_scores.append(homogeneity_score(y, labels))

# Plotando a Pontuação Silhouette e Inércia
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(k_values, silhouette_scores, 'g-')
ax2.plot(k_values, inertias, 'b-')

ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Pontuação Silhouette', color='g')
ax2.set_ylabel('Inércia', color='b')

plt.title('Silhouette e Inércia para Diferentes Valores de k')
plt.show()

# Plotando Homogeneidade
plt.figure(figsize=(10, 5))
plt.plot(k_values, homogeneity_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Homogeneidade')
plt.title('Homogeneidade para Diferentes Valores de k')
plt.show()

# Visualização dos clusters formados para diferentes valores de k
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(f'Clusters Formados pelo KMeans com k={k} (PCA)')
    plt.xlabel('Componente PCA 1')
    plt.ylabel('Componente PCA 2')
    plt.colorbar(label='Clusters')
    plt.show()
