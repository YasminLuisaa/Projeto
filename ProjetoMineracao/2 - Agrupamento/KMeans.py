import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Definindo nossa função KMeans do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Escolhendo centroides aleatórios
    centroids = x[idx, :]  # Passo 1
     
    # Encontrando a distância entre os centroides e todos os pontos de dados
    distances = cdist(x, centroids, 'euclidean')  # Passo 2
     
    # Centróide com a menor distância
    points = np.array([np.argmin(i) for i in distances])  # Passo 3
     
    # Repetindo os passos acima por um número definido de iterações
    # Passo 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            # Atualizando os centroides tomando a média do cluster a que pertencem
            temp_cent = x[points == idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids)  # Centroides atualizados
         
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points

def plot_samples(projected, labels, title):    
    plt.figure(figsize=(10, 7))
    plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Componente PCA 1')
    plt.ylabel('Componente PCA 2')
    plt.colorbar(label='Clusters')
    plt.show()

def main():
    # Carregar dataset Digits
    digits = load_digits()
    
    # Transformar os dados usando PCA
    pca = PCA(2)
    projected = pca.fit_transform(digits.data)
    print(pca.explained_variance_ratio_)
    print(digits.data.shape)
    print(projected.shape)    
    plot_samples(projected, digits.target, 'Original Labels')
 
    # Aplicando nossa função kmeans do zero
    labels_manual = KMeans_scratch(projected, 2, 5)
    
    # Aplicar KMeans com sklearn
    kmeans = KMeans(n_clusters=2).fit(projected)
    print(kmeans.inertia_)
    centers = kmeans.cluster_centers_
    score = silhouette_score(projected, kmeans.labels_)    
    print(f"For n_clusters = {2}, silhouette score is {score}")

    # Ajustar os rótulos manuais para coincidir com os do sklearn, se necessário
    if np.sum(labels_manual == kmeans.labels_) < np.sum(labels_manual != kmeans.labels_):
        labels_manual = 1 - labels_manual  # Inverte 0 -> 1 e 1 -> 0

    # Visualizar os resultados da implementação manual
    plot_samples(projected, labels_manual, 'Clusters Labels KMeans from scratch (Adjusted)')

    # Visualizar os resultados do sklearn
    plot_samples(projected, kmeans.labels_, 'Clusters Labels KMeans from sklearn')

if __name__ == "__main__":
    main()
