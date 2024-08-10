import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para carregar a base de dados spambase
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
    print("Total samples: {}".format(X.shape[0]))

    # Escalar os dados X usando Z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Definir o número de folds para cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)

    # Implementando KNN com cross-validation
    knn = KNeighborsClassifier(n_neighbors=5)
    scores_knn = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    print("Accuracy KNN (Cross-Validation): {:.2f}% ± {:.2f}%".format(scores_knn.mean() * 100, scores_knn.std() * 100))

    # Implementando SVM com cross-validation
    svm = SVC(kernel='poly')
    scores_svm = cross_val_score(svm, X, y, cv=kf, scoring='accuracy')
    print("Accuracy SVM (Cross-Validation): {:.2f}% ± {:.2f}%".format(scores_svm.mean() * 100, scores_svm.std() * 100))

    # Implementando Árvore de Decisão com cross-validation
    dt = DecisionTreeClassifier()
    scores_dt = cross_val_score(dt, X, y, cv=kf, scoring='accuracy')
    print("Accuracy Decision Tree (Cross-Validation): {:.2f}% ± {:.2f}%".format(scores_dt.mean() * 100, scores_dt.std() * 100))

    # Implementando Rede Neural com cross-validation
    def build_model():
        model = Sequential()
        model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    scores_nn = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = build_model()
        model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=0)
        y_hat_test_nn = (model.predict(X_test) > 0.5).astype("int32")
        scores_nn.append(accuracy_score(y_test, y_hat_test_nn))

    scores_nn = np.array(scores_nn)
    print("Accuracy Neural Network (Cross-Validation): {:.2f}% ± {:.2f}%".format(scores_nn.mean() * 100, scores_nn.std() * 100))

if __name__ == "__main__":
    main()
