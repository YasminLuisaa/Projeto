# Importações iniciais
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Esta função imprime e plota a matriz de confusão.
    A normalização pode ser aplicada definindo `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão sem normalização')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')

def load_spambase_dataset(file_path):    
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
    data = pd.read_csv(file_path, names=column_names)
    return data

def main():
    # Carregar os dados spambase
    input_file = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'
    data = load_spambase_dataset(input_file)

    # Separar os dados em X e y
    X = data.drop('spam', axis=1)
    y = data['spam']
    target_names = ['not spam', 'spam']
    print("Total samples: {}".format(X.shape[0]))

    # Dividir os dados - 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Escalar os dados X usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TESTES USANDO CLASSIFICADOR SVM DO SK-LEARN
    svm = SVC(kernel='poly')  # poly, rbf, linear
    # Treinando usando o conjunto de dados de treino
    svm.fit(X_train, y_train)
    # Obter vetores de suporte
    print(svm.support_vectors_)
    # Obter índices dos vetores de suporte
    print(svm.support_)
    # Obter número de vetores de suporte para cada classe
    print("Qtd de vetores de suporte: ")
    print(svm.n_support_)
    # Prever usando o conjunto de dados de teste
    y_hat_test = svm.predict(X_test)

    # Obter a precisão das previsões
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}%".format(f1))

    # Obter a matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - SVM sklearn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - SVM sklearn normalized")
    plt.show()

if __name__ == "__main__":
    main()
