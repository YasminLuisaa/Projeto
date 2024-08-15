import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import itertools

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('int')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

def build_neural_network():
    model = Sequential()
    model.add(Dense(64, input_dim=57, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_model_with_cross_validation(X_train, y_train, X_test, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Accuracy (Holdout): {accuracy:.4f}")
    print(f"{model_name} F1 Score (Holdout): {f1:.4f}")
    print(f"{model_name} Precision (Holdout): {precision:.4f}")
    print(f"{model_name} Recall (Holdout): {recall:.4f}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    
    print(f"{model_name} Cross-validation Accuracy: {cv_accuracy:.4f}")
    print(f"{model_name} Cross-validation F1 Score: {cv_f1:.4f}")
    print(f"{model_name} Cross-validation Precision: {cv_precision:.4f}")
    print(f"{model_name} Cross-validation Recall: {cv_recall:.4f}")
    
    plot_confusion_matrix(cm, ['not spam', 'spam'], f"Confusion Matrix - {model_name}")
    
    return accuracy, f1, precision, recall, cm, cv_accuracy, cv_f1, cv_precision, cv_recall

def evaluate_neural_network_holdout(X_train, y_train, X_test, y_test):
    model = build_neural_network()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Neural Network Accuracy (Holdout): {accuracy:.4f}")
    print(f"Neural Network F1 Score (Holdout): {f1:.4f}")
    print(f"Neural Network Precision (Holdout): {precision:.4f}")
    print(f"Neural Network Recall (Holdout): {recall:.4f}")

    plot_confusion_matrix(cm, ['not spam', 'spam'], "Confusion Matrix - Neural Network (Holdout)")

    return accuracy, f1, precision, recall, cm

def main():
    input_file = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'
    data = load_spambase_dataset(input_file)

    X = data.drop('spam', axis=1)
    y = data['spam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    evaluate_model_with_cross_validation(X_train, y_train, X_test, y_test, knn, "KNN")

    # SVM
    svm = SVC(kernel='rbf')
    evaluate_model_with_cross_validation(X_train, y_train, X_test, y_test, svm, "SVM")

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10)
    evaluate_model_with_cross_validation(X_train, y_train, X_test, y_test, dt, "Decision Tree")

    # Neural Network with Holdout
    evaluate_neural_network_holdout(X_train, y_train, X_test, y_test)

    plt.show()

if __name__ == "__main__":
    main()
