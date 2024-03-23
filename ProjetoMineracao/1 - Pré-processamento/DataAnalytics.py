import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Define o caminho do arquivo CSV
    arquivo_entrada = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'

    # Lista com os nomes das colunas
    nomes = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
             'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
             'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
             'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
             'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
             'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
             'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
             'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
             'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
             'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
             'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
             'capital_run_length_longest', 'capital_run_length_total', 'spam']

    # Carrega o arquivo CSV em um DataFrame do Pandas
    df = pd.read_csv(arquivo_entrada, names=nomes)

    # Seleciona apenas as características específicas para análise
    selected_features = ['word_freq_free', 'char_freq_$']

    # Loop sobre cada característica selecionada para análise
    for feature in selected_features:
        # Medidas de tendência central
        print('+++ Medidas de tendência central +++')
        print(f"Nome do atributo: {feature}")
        print(f"Média: {df[feature].mean()}")
        print(f"Mediana: {df[feature].median()}")
        print(f"Ponto Médio: {(df[feature].max() + df[feature].min()) / 2}")
        print(f"Moda: {df[feature].mode().values[0]}")
        print('\n')

        # Medidas de dispersão
        print(f'+++ Medidas de dispersão +++')
        print(f"Nome do atributo: {feature}")
        print(f"Amplitude: {df[feature].max() - df[feature].min()}")
        print(f"Desvio Padrão: {df[feature].std()}")
        print(
            f"Desvio Absoluto: {df[feature].apply(lambda x: abs(x - df[feature].median())).median()}")
        print(f"Variância: {df[feature].var()}")
        print(
            f"Coeficiente de Variação: {(df[feature].std() / df[feature].mean()) * 100:.2f}%")
        print("\n")

        # Medidas de forma (Curtose)
        print(f'+++ Medidas de forma (Curtose) +++')
        print(f"Curtose: {df[feature].kurtosis()}")
        print("\n")

        # Medidas de posição relativa
        print(f'+++ Medidas de posição relativa +++')
        print(
            f"Escore Z:\n {(df[feature] - df[feature].mean()) / df[feature].std()}")
        print("\n")
        print(f"Quantis:\n {df[feature].quantile([0.25, 0.5, 0.75])}")
        print("\n")

        # Medidas de associação
        print(f'+++ Medidas de associação +++')
        print(f"Covariância com 'spam': {df[feature].cov(df['spam'])}")
        print(f"Correlação com 'spam': {df[feature].corr(df['spam'])}")
        print("\n")

    # Configura o estilo do gráfico usando o seaborn
    sns.set_style("whitegrid")

    # Plota o gráfico de barras dos valores médios das características selecionadas
    plt.figure(figsize=(10, 6))
    df[selected_features].mean().plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Valores Médios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Para cada característica selecionada, cria um gráfico de dispersão
    for feature in selected_features:
        create_scatterplot(df, 'spam', feature)


def create_scatterplot(df, target_variable, selected_feature):
    # Cria um gráfico de dispersão
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x=target_variable, y=selected_feature,
                         hue=target_variable, palette='tab10', legend=False)
    ax.set_title(selected_feature)
    ax.set_xlabel('spam')
    ax.set_ylabel(selected_feature)
    plt.show()


if __name__ == "__main__":
    main()
