import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Carregar dados (assumindo que você já fez isso)
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

# Carregar dados
spambase_data = pd.read_csv(input_file, header=None)
spambase_data.columns = column_names

# Selecionar características mais relevantes 
top_features = ['word_freq_your', 'word_freq_000', 'word_freq_remove', 'char_freq_$', 'char_freq_!']

# Dados reduzidos com as características mais relevantes
X_reduced = spambase_data[top_features]

# Normalização dos dados
scaler = StandardScaler()
X_reduced_scaled = scaler.fit_transform(X_reduced)

# Redução de dimensionalidade com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_reduced_scaled)

# Aplicação do KMeans com 2 clusters (melhor número encontrado)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X_pca)
kmeans_silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
print(f"Pontuação Silhouette do KMeans: {kmeans_silhouette_avg}")

# Visualização dos clusters com PCA
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
plt.title('Clusters Formados pelo KMeans com PCA')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.colorbar(label='Clusters')
plt.show()

# Centroides dos clusters
centroids = scaler.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))
centroid_df = pd.DataFrame(centroids, columns=top_features)
print("Centroides dos clusters:")
print(centroid_df)
