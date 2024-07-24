import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carregar dados
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

# Carregar os dados
spambase_data = pd.read_csv(input_file, names=column_names)

# Seleção de características
correlation_matrix = spambase_data.corr()
top_features = correlation_matrix['spam'].abs().sort_values(ascending=False).index[1:11]

# Dados reduzidos com as características mais relevantes
X_reduced = spambase_data[top_features]
y = spambase_data['spam']

# Normalização dos dados
scaler = StandardScaler()
X_reduced_scaled = scaler.fit_transform(X_reduced)

# Holdout (70% Treinamento, 30% Teste)
X_train, X_test, y_train, y_test = train_test_split(X_reduced_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Classificação com K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Métricas de Holdout
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Matriz de Confusão (Holdout):")
print(conf_matrix)
print(f"Acurácia (Holdout): {accuracy}")
print(f"F1 Score (Holdout): {f1}")

# Cross-Validation (k=10)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(knn, X_reduced_scaled, y, cv=cv, scoring='accuracy')
cv_f1 = cross_val_score(knn, X_reduced_scaled, y, cv=cv, scoring='f1')

print(f"Acurácia Média (Cross-Validation): {cv_accuracy.mean()}")
print(f"F1 Score Médio (Cross-Validation): {cv_f1.mean()}")

# Relatório de Classificação (Holdout)
print("\nRelatório de Classificação (Holdout):")
print(classification_report(y_test, y_pred))

# Visualização da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão (Holdout)')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()
