import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import itertools

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Carregar a base de dados spambase
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

    # Implementando KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_hat_test_knn) * 100
    f1_knn = f1_score(y_test, y_hat_test_knn, average='macro')
    cm_knn = confusion_matrix(y_test, y_hat_test_knn)
    print("Accuracy KNN: {:.2f}%".format(accuracy_knn))
    print("F1 Score KNN: {:.2f}".format(f1_knn))
    plot_confusion_matrix(cm_knn, target_names, False, "Confusion Matrix - KNN")

    # Implementando SVM
    svm = SVC(kernel='poly')
    svm.fit(X_train, y_train)
    y_hat_test_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_hat_test_svm) * 100
    f1_svm = f1_score(y_test, y_hat_test_svm, average='macro')
    cm_svm = confusion_matrix(y_test, y_hat_test_svm)
    print("Accuracy SVM: {:.2f}%".format(accuracy_svm))
    print("F1 Score SVM: {:.2f}".format(f1_svm))
    plot_confusion_matrix(cm_svm, target_names, False, "Confusion Matrix - SVM")

    # Implementando Árvore de Decisão
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_hat_test_dt = dt.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_hat_test_dt) * 100
    f1_dt = f1_score(y_test, y_hat_test_dt, average='macro')
    cm_dt = confusion_matrix(y_test, y_hat_test_dt)
    print("Accuracy Decision Tree: {:.2f}%".format(accuracy_dt))
    print("F1 Score Decision Tree: {:.2f}".format(f1_dt))
    plot_confusion_matrix(cm_dt, target_names, False, "Confusion Matrix - Decision Tree")

    # Implementando Rede Neural com Arquitetura Melhorada
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Saída para classificação binária

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar a rede neural
    history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.2, verbose=1)

    # Avaliar a rede neural
    y_hat_test_nn = (model.predict(X_test) > 0.5).astype("int32")

    # Calcular métricas
    accuracy_nn = accuracy_score(y_test, y_hat_test_nn) * 100
    f1_nn = f1_score(y_test, y_hat_test_nn, average='macro')
    cm_nn = confusion_matrix(y_test, y_hat_test_nn)
    print("Accuracy Neural Network: {:.2f}%".format(accuracy_nn))
    print("F1 Score Neural Network: {:.2f}".format(f1_nn))
    plot_confusion_matrix(cm_nn, target_names, False, "Confusion Matrix - Neural Network")
    plot_confusion_matrix(cm_nn, target_names, True, "Confusion Matrix - Neural Network normalized")

    plt.show()

if __name__ == "__main__":
    main()
