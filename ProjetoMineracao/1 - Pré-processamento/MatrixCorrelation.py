import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Carregar os dados
    arquivo_entrada = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'
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
    
    df = pd.read_csv(arquivo_entrada, names=nomes) 

    # Calcular a matriz de correlação
    matriz_correlacao = df.corr()

    # Plotar a matriz de correlação sem os valores
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz_correlacao, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('Matriz_de_Correlação.png', format='png')
    plt.show()
    
    # Plotar a matriz de correlação com valores
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('Matriz_de_Correlação.png', format='png')
    plt.show()


if __name__ == "__main__":
    main()