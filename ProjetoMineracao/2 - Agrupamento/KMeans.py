import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

# Selecionando os dados
X = spambase_data.drop('spam', axis=1)
y = spambase_data['spam']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Instanciar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Validação cruzada
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
print(f"Validação Cruzada - Acurácia média: {cv_scores.mean():.2f}")

# Treinar o modelo
knn.fit(X_train, y_train)

# Predições
y_pred = knn.predict(X_test)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.show()

# Métricas de Classificação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Revocação:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Relatório de Classificação completo
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
