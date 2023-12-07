# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('starbucks.csv',sep=';')
df.columns = df.columns.str.strip()

#print(df.columns)

# Calcular medidas de tendencia central por 'Beverage_category'
grouped_data = df.groupby('Beverage_category')['Calories'].agg(['mean', 'median', 'std']).reset_index()

# Visualizar gráficamente con un diagrama de cajas y bigotes
plt.figure(figsize=(12, 8))
sns.boxplot(x='Beverage_category', y='Calories', data=df)
plt.title('Boxplot de Calories por Beverage_category')
plt.show()

# Mostrar las medidas de tendencia central
print(grouped_data)

# Histograma de frecuencias para CALORIAS
plt.figure(figsize=(10, 6))
sns.histplot(df['Calories'], bins=20, color='blue', kde=True, edgecolor='black')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.title('Histogram of Calories Distribution with KDE')
plt.show()

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop(['Calories', 'Beverage_category', 'Beverage', 'Beverage_prep'], axis=1)
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Modelo de regresión logística
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Modelo de Árboles de decisión y clasificación (CART)
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Calcular indicadores para comparar los modelos
linear_rmse = mean_squared_error(y_test, linear_predictions, squared=False)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)

# Imprimir los resultados
print(f'Regresión Lineal RMSE: {linear_rmse}')
print(f'Regresión Logística Accuracy: {logistic_accuracy}')
print(f'Árboles de Decisión Accuracy: {tree_accuracy}')

# Puedes también imprimir otros indicadores como el reporte de clasificación para el modelo logístico
print('\nReporte de Clasificación para Regresión Logística:')
print(classification_report(y_test, logistic_predictions))
