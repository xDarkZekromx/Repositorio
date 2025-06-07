#recommendation_system.py

#importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors 

#cargar el dataset
#este dataset debe contener al menos dos columnas: 'user_id' y 'item_id'

data = pd.read_csv('dataset.csv')

#preprocesar los datos
#seleccionar las características relevantes
features = data[['user_id', 'item_id']]
labels = data['rating']  # o cualquier otra columna que represente la interacción
#dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#entrenar el modelo de recomendación
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(X_train)
#hacer predicciones
distances, indices = model.kneighbors(X_test, n_neighbors=5)
#evaluar el modelo
y_pred = np.mean(y_train.iloc[indices], axis=1)
#calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
#imprimir las métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
#guardar el modelo entrenado
import joblib
joblib.dump(model, 'recommendation_model.pkl')
#cargar el modelo guardado
loaded_model = joblib.load('recommendation_model.pkl')
#hacer recomendaciones para un usuario específico
user_id = 1  # Cambia esto al ID del usuario para el que quieres hacer recomendaciones
user_items = data[data['user_id'] == user_id]['item_id'].values
distances, indices = loaded_model.kneighbors(user_items.reshape(1, -1), n_neighbors=5)
#mostrar las recomendaciones
recommended_items = data.iloc[indices[0]]['item_id'].values
print(f'Recommended items for user {user_id}: {recommended_items}')
print("Recomendaciones generadas con éxito.")
#guardar las recomendaciones en un archivo
#fin del script