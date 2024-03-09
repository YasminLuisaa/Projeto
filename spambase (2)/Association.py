import pandas as pd

df = pd.read_csv('C:\\Users\\CASA\\Documents\\spambase\\spambase.data')


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)
