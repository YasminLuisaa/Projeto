import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    arquivo_entrada = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'  # Substitua pelo caminho real para o seu arquivo de dados spambase
    nomes = ['word_freq_WORD_' + str(i) for i in range(48)] + \
            ['char_freq_CHAR_' + str(i) for i in range(6)] + \
            ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']
    df = pd.read_csv(arquivo_entrada, names=nomes)

    # Exibe informações sobre o dataframe original
    MostraInformacoesDataFrame(df, "DataFrame Original")

    # Separando as características
    features = df.columns[:-1]
    target = 'spam'
    x = df[features].values

    # Separando o alvo
    y = df[target].values

    # Projeção PCA
    pca = PCA()
    componentes_principais = pca.fit_transform(x)
    print("Variância explicada por componente:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")
    
    # Cria um novo dataframe contendo as duas primeiras componentes principais e o alvo
    df_principal = pd.DataFrame(data=componentes_principais[:, 0:2],
                                columns=['Componente Principal 1', 'Componente Principal 2'])
    df_final = pd.concat([df_principal, df[[target]]], axis=1)
    # Exibe informações sobre o dataframe resultante após a PCA
    MostraInformacoesDataFrame(df_final, "DataFrame PCA")
    
    # Visualiza a projeção PCA em um gráfico de dispersão
    VisualizarProjecaoPca(df_final, target)
    
# Função para exibir informações sobre um dataframe
def MostraInformacoesDataFrame(df, mensagem=""):
    print(mensagem + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

# Função para visualizar a projeção PCA em um gráfico de dispersão
def VisualizarProjecaoPca(df_final, coluna_alvo):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Componente Principal 1', fontsize=15)
    ax.set_ylabel('Componente Principal 2', fontsize=15)
    ax.set_title('Projeção PCA de 2 componentes', fontsize=20)
    alvos = [0, 1]
    cores = ['r', 'g']
    for alvo, cor in zip(alvos, cores):
        indicesManter = df_final[coluna_alvo] == alvo
        ax.scatter(df_final.loc[indicesManter, 'Componente Principal 1'],
                   df_final.loc[indicesManter, 'Componente Principal 2'],
                   c=cor, s=50)
    ax.legend(alvos)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
