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

    # Histograma para a coluna char_freq_$
    plt.figure(figsize=(8, 6))
    plt.hist(df['char_freq_$'], bins=20, color='blue', alpha=0.5)
    plt.title('Histograma para a coluna char_freq_$')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.savefig('Histograma.png', format='png')
    plt.show()

    # Gráfico de barra para spam e não-spam
    spam_counts = df['spam'].value_counts()
    plt.figure(figsize=(8, 6))
    spam_counts.plot(kind='bar', color='blue', alpha=0.5)
    plt.title('Contagem de spam e não-spam')
    plt.xlabel('Categoria')
    plt.ylabel('Contagem')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Barra.png', format='png')
    plt.show()

    # Gráfico de pizza para spam e não-spam
    contagem_spam = df['spam'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(contagem_spam, labels=contagem_spam.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('Proporção de não spam e spam')
    plt.tight_layout()
    plt.savefig('Pizza.png', format='png')
    plt.show()

    # Gráfico de Dispersão para char_freq_! vs word_freq_money
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='char_freq_!', y='word_freq_money', data=df)
    plt.title('Gráfico de Dispersão: char_freq_! vs word_freq_money')
    plt.xlabel('char_freq_!')
    plt.ylabel('word_freq_money')
    plt.tight_layout()
    plt.savefig('Gráfico_de_Dispersão.png', format='png')
    plt.show()

    # Diagrama de caixa para word_freq_make vs spam
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='spam', y='word_freq_make', data=df)
    plt.title('Diagrama de caixa: word_freq_make vs spam')
    plt.xlabel('spam')
    plt.ylabel('word_freq_make')
    plt.tight_layout()
    plt.savefig('BoxPlot.png', format='png')
    plt.show()
    
if __name__ == "__main__":
    main()
