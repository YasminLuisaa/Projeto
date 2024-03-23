import pandas as pd
import numpy as np

def principal():
    # Atualize os caminhos dos arquivos de entrada e saída
    arquivo_entrada = 'C:\\Users\\CASA\Downloads\spambase (2)\\ProjetoMineracao\\0 - Conjunto de dados\\spambase.data'
    arquivo_saida = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_processado.csv'

    # Defina os nomes das colunas e características com base no conjunto de dados de spam
    nomes = [
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
        'char_freq_semicolon', 'char_freq_leftparen', 'char_freq_leftsquare',
        'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
        'capital_run_length_average', 'capital_run_length_longest',
        'capital_run_length_total', 'spam'
    ]

    caracteristicas = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
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
        'char_freq_semicolon', 'char_freq_leftparen', 'char_freq_leftsquare',
        'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
        'capital_run_length_average', 'capital_run_length_longest',
        'capital_run_length_total', 'spam']  
    
    # Leia o conjunto de dados de spam
    df = pd.read_csv(arquivo_entrada, names=nomes, usecols=caracteristicas, na_values='?')

    # Copie o dataframe original
    df_original = df.copy()

    # Imprima informações sobre os dados
    print("INFORMAÇÕES GERAIS SOBRE OS DADOS\n")
    print(df.info())
    print("\n")

    # Imprima análise descritiva dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")

    # Imprima o número de valores ausentes por coluna
    print("VALORES AUSENTES\n")
    print(df.isnull().sum())
    print("\n")

    colunas_valores_ausentes = df.columns[df.isnull().any()]

    # Escolha um método para lidar com valores ausentes (por exemplo, 'mode' para preenchimento com moda)
    metodo = 'mode'

    for c in colunas_valores_ausentes:
        AtualizarValoresAusentes(df, c, metodo)

    # Imprima a análise descritiva atualizada dos dados
    print("DESCRIÇÃO DOS DADOS ATUALIZADA\n")
    print(df.describe())
    print("\n")

    # Salve os dados processados em um novo arquivo
    df.to_csv(arquivo_saida, header=False, index=False)

def AtualizarValoresAusentes(df, coluna, metodo="mode"):
    if metodo == 'number':
        # Substitua os valores ausentes por um número específico
        df[coluna].fillna(0, inplace=True)
    elif metodo == 'median':
        # Substitua os valores ausentes pela mediana
        mediana = df[coluna].median()
        df[coluna].fillna(mediana, inplace=True)
    elif metodo == 'mean':
        # Substitua os valores ausentes pela média
        media = df[coluna].mean()
        df[coluna].fillna(media, inplace=True)
    elif metodo == 'mode':
        # Substitua os valores ausentes pela moda
        moda = df[coluna].mode()[0]
        df[coluna].fillna(moda, inplace=True)


if __name__ == "__main__":
    principal()
