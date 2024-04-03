import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

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

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(df.drop('spam', axis=1), df['spam'], test_size=0.2, random_state=42)

    # Reamostragem usando SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Ajuste de peso de classe
    class_weights = dict({0: 1, 1: len(y_train) / (2 * sum(y_train))})

    # Inicializar e treinar o modelo Random Forest com os dados reamostrados e pesos de classe ajustados
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Fazer previsões no conjunto de teste
    y_pred = rf_model.predict(X_test)

    # Avaliar o desempenho do modelo
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
