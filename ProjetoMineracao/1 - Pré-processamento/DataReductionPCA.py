# Main function
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Path to the input file
    input_file = 'C:\\Users\\CASA\\Downloads\\spambase (2)\\ProjetoMineracao\\1 - Pré-processamento\\spam_data_normalizado.csv'
    
    # Column names of the dataset
    column_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
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
    
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(input_file, names=column_names)

    # Displaying information about the original DataFrame
    show_dataframe_info(df, "Original DataFrame")

    # Selecting features and target
    features = df.columns[:-1]
    target = 'spam'
    x = df[features].values
    y = df[target].values

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # PCA 2D
    visualize_pca_2d(x_scaled, y, features)

    # PCA 3D
    visualize_pca_3d(x_scaled, y, features)

    # t-SNE 2D
    visualize_tsne_2d(x_scaled, y, features)

# Function to display information about a DataFrame
def show_dataframe_info(df, message=""):
    print(message + "\n")
    print(df.info())        # General information about the DataFrame
    print(df.describe())    # Descriptive statistics of the DataFrame
    print(df.head(10))      # Displaying the first few rows of the DataFrame
    print("\n")

    # Display minimum and maximum values of each feature
    print("Minimum Values of Features:")
    print(df.min())
    print("\nMaximum Values of Features:")
    print(df.max())
    print("\n")

# Function to visualize PCA in 2D
def visualize_pca_2d(x, y, features):
    # PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)

    # Plotting PCA 2D results
    plt.figure(figsize=(8, 6))
    plot_scatter(pca_result, y, title='PCA 2D')
    plt.show()

# Function to visualize PCA in 3D
def visualize_pca_3d(x, y, features):
    # PCA 3D
    pca = PCA(n_components=3)
    pca_result_3d = pca.fit_transform(x)

    # Plotting PCA 3D results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_scatter_3d(pca_result_3d, y, ax, title='PCA 3D')
    plt.show()

# Function to visualize t-SNE in 2D
def visualize_tsne_2d(x, y, features):
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(x)

    # Plotting t-SNE 2D results
    plt.figure(figsize=(8, 6))
    plot_scatter(tsne_result, y, title='t-SNE 2D')
    plt.show()

# Function to plot 2D scatter plot
def plot_scatter(data, y, title):
    plt.scatter(data[y == 0, 0], data[y == 0, 1], c='r', label='Não Spam', alpha=0.5)
    plt.scatter(data[y == 1, 0], data[y == 1, 1], c='g', label='Spam', alpha=0.5)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title(title)
    plt.legend()
    plt.grid()

# Function to plot 3D scatter plot for PCA
def plot_scatter_3d(data, y, ax, title):
    ax.scatter(data[y == 0, 0], data[y == 0, 1], data[y == 0, 2], c='r', label='Não Spam', alpha=0.5)
    ax.scatter(data[y == 1, 0], data[y == 1, 1], data[y == 1, 2], c='g', label='Spam', alpha=0.5)
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    ax.set_title(title)
    ax.legend()
    ax.grid()

# Executing the main function
if __name__ == "__main__":
    main()
