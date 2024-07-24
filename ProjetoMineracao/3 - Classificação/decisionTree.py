import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

def main():
    # Carregar dados do spambase
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
    spambase_data = pd.read_csv(input_file, header=None)
    spambase_data.columns = column_names

    # Preparação dos dados
    X = spambase_data.drop(['spam'], axis=1)
    y = spambase_data['spam']

    # Divisão em conjunto de treino e teste - 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Treinamento do modelo de árvore de decisão
    clf = DecisionTreeClassifier(max_leaf_nodes=3)
    clf.fit(X_train, y_train)

    # Previsões nos conjuntos de treino e teste
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Cálculo da acurácia
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Acurácia no conjunto de treino: {train_accuracy:.2f}")
    print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

    # Plotar a árvore de decisão
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Non-SPAM', 'SPAM'])
    plt.title('Decision Tree for SPAM Classification')
    plt.show()

if __name__ == "__main__":
    main()
