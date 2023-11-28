# -*- coding: utf-8 -*-
"""Proyecto Final ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TBbDKe-iv7-UQhWB-dijx778s_Gkpqq5

#Transformación de dataset
"""

import pandas as pd

# Cargar el dataset
df = pd.read_csv('weather.csv',encoding='latin-1')
print(df.head())
# Reemplazar valores nulos en columnas específicas con la media
columns_to_fillna = ['Temperatura_Mínima','Temperatura_Máxima','Precipitación','Evaporación','Horas_de_Sol','Velocidad_ráfaga_viento', 'Velocidad_viento_9am','Velocidad_viento_3pm',
                     'Humedad_9am', 'Humedad_3pm', 'Presión_9am', 'Presión_3pm', 'Nubosidad_9am', 'Nubosidad_3pm',
                     'Temperatura_9am', 'Temperatura_3pm']
df[columns_to_fillna] = df[columns_to_fillna].fillna(df[columns_to_fillna].mean())

# Convertir la columna de fecha a tipo datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

# Eliminar duplicados
df = df.drop_duplicates()

# Codificar variables categóricas
df = pd.get_dummies(df, columns=['Ubicación', 'Dirección_ráfaga_viento', 'Dirección_viento_9am', 'Dirección_viento_3pm'])

# Eliminar columnas no necesarias
columns_to_drop = ['Evaporación']  # Agrega aquí las columnas que deseas eliminar
df = df.drop(columns=columns_to_drop, axis=1)

# Normalizar variables (por ejemplo, utilizando Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Temperatura_Mínima', 'Temperatura_Máxima', 'Precipitación', 'Horas_de_Sol',
    'Velocidad_ráfaga_viento', 'Velocidad_viento_9am', 'Velocidad_viento_3pm',
    'Humedad_9am', 'Humedad_3pm', 'Presión_9am', 'Presión_3pm', 'Nubosidad_9am',
    'Nubosidad_3pm', 'Temperatura_9am', 'Temperatura_3pm']] = scaler.fit_transform(df[['Temperatura_Mínima', 'Temperatura_Máxima', 'Precipitación', 'Horas_de_Sol',
    'Velocidad_ráfaga_viento', 'Velocidad_viento_9am', 'Velocidad_viento_3pm',
    'Humedad_9am', 'Humedad_3pm', 'Presión_9am', 'Presión_3pm', 'Nubosidad_9am',
    'Nubosidad_3pm', 'Temperatura_9am', 'Temperatura_3pm']])


# Visualizar las primeras filas del dataset limpio
print(df.head())

# Guardar el dataset limpio
df.to_csv('weather_clean.csv', index=False)

!pip install unidecode
from unidecode import unidecode

# Cargar el conjunto de datos
df = pd.read_csv('weather_clean.csv')  # Asegúrate de tener el nombre correcto del archivo

# Quitar acentos de los nombres de las columnas
df.columns = [unidecode(col) for col in df.columns]

# Visualizar las primeras filas del conjunto de datos con los nombres de columnas actualizados
print(df.head())

# Guardar el conjunto de datos con los nombres de columnas actualizados
df.to_csv('weather_clean_no_accent.csv', index=False)

"""#EDA"""

!pip install pyspark
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WeatherAnalysis").getOrCreate()
df = spark.read.csv("weather_clean.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("weather_data")

"""Temperatura máxima y mínima"""

result_avg_temp = spark.sql("SELECT AVG(Temperatura_Minima) AS avg_temp_min, AVG(Temperatura_Maxima) AS avg_temp_max FROM weather_data")
result_avg_temp.show()

"""Total de días lluviosos"""

result_rainy_days = spark.sql("SELECT COUNT(*) AS rainy_days FROM weather_data WHERE Lluvia_hoy = 'Yes'")
result_rainy_days.show()

"""Días más cálidos y más fríos:"""

result_hottest_days = spark.sql("SELECT Fecha, Temperatura_Maxima FROM weather_data ORDER BY Temperatura_Maxima DESC LIMIT 5")
result_coldest_days = spark.sql("SELECT Fecha, Temperatura_Minima FROM weather_data ORDER BY Temperatura_Minima ASC LIMIT 5")
result_hottest_days.show()
result_coldest_days.show()

"""Días con mayor y menor precipitación:"""

result_rainiest_days = spark.sql("SELECT Fecha, Precipitacion FROM weather_data ORDER BY Precipitacion DESC LIMIT 5")
result_driest_days = spark.sql("SELECT Fecha, Precipitacion FROM weather_data WHERE Precipitacion > 0 ORDER BY Precipitacion ASC LIMIT 5")
result_rainiest_days.show()
result_driest_days.show()

"""Días con cambios bruscos en la velocidad del viento:

"""

result_wind_change = spark.sql("SELECT Fecha, (Velocidad_viento_3pm - Velocidad_viento_9am) AS wind_change FROM weather_data ORDER BY wind_change DESC LIMIT 5")
result_wind_change.show()

"""Días con mayor cambio de temperatura"""

result_temp_change = spark.sql("SELECT Fecha, (Temperatura_Maxima - Temperatura_Minima) AS temp_change FROM weather_data ORDER BY temp_change DESC LIMIT 5")
result_temp_change.show()

"""Días con Horas de Sol Bajas"""

query = spark.sql("""
SELECT COUNT(*) AS Low_Sunlight_Days FROM weather_data WHERE Horas_de_Sol < 6;
""")
query.show()

"""# Comparación entre modelos

## Importación de librerías para el proyecto
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

"""## Preprocesamiento de Datos y División en Conjuntos de Entrenamiento y Prueba"""

# Cargar los datos

df = pd.read_csv('/weather.csv')

# Conversión de columnas de fecha
df['C_1'] = pd.to_datetime(df['C_1'])
df['Año'] = df['C_1'].dt.year
df['Mes'] = df['C_1'].dt.month
df['Día'] = df['C_1'].dt.day

# Limpieza de datos: Rellenar valores faltantes
df.fillna(method='ffill', inplace=True)

# Identificar columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('C_23')  # Asumiendo que 'C_23' es la columna objetivo

# Preprocesamiento de columnas
preprocessorLR = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Preprocesamiento de columnas
preprocessorRF = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Preprocesamiento de columnas
preprocessorDT = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Convertir la columna objetivo a valores binarios
df['C_23'] = df['C_23'].map({'Yes': 1, 'No': 0})

# Separar las características y el objetivo
X = df.drop('C_23', axis=1)
y = df['C_23']

# Dividir en conjuntos de entrenamiento y prueba para cada modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Construcción de Pipelines y entrenamiento de los modelos"""

# Crear y ajustar el pipeline de regresión logística
pipelineLR = Pipeline(steps=[('preprocessor', preprocessorLR),
                           ('classifier', LogisticRegression(max_iter=1000, C=1.0))])


# Entrenar el modelo de regresión logística
pipelineLR.fit(X_train, y_train)

# Crear y ajustar el pipeline de Ramdom Forest
pipelineRF = Pipeline(steps=[('preprocessor', preprocessorRF),
                           ('classifier', RandomForestClassifier())])


# Entrenar el modelo de Ramdom Forest
pipelineRF.fit(X_train1, y_train1)

# Crear y ajustar el pipeline de árbol de decisión
pipelineDT = Pipeline(steps=[('preprocessor', preprocessorDT),
                           ('classifier', RandomForestClassifier())])


# Entrenar el modelo de árbol de decisión
pipelineDT.fit(X_train2, y_train2)

"""## Evaluación de Modelos"""

# Predecir en el conjunto de prueba
y_pred = pipelineLR.predict(X_test)

# Calcular métricas
accuracyLR = accuracy_score(y_test, y_pred)
recallLR = recall_score(y_test, y_pred)
f1LR = f1_score(y_test, y_pred)
roc_aucLR = roc_auc_score(y_test, y_pred)

# Mostrar métricas
print('Logistic Regression')
print(f'Accuracy: {accuracyLR}')
print(f'Recall: {recallLR}')
print(f'F1 Score: {f1LR}')
print(f'ROC AUC: {roc_aucLR}')
print('\n')

# Predecir en el conjunto de prueba
y_pred1 = pipelineRF.predict(X_test1)

# Calcular métricas
accuracyRF = accuracy_score(y_test1, y_pred1)
recallRF = recall_score(y_test1, y_pred1)
f1RF = f1_score(y_test1, y_pred1)
roc_aucRF = roc_auc_score(y_test1, y_pred1)

# Mostrar métricas
print('Random Forest')
print(f'Accuracy: {accuracyRF}')
print(f'Recall: {recallRF}')
print(f'F1 Score: {f1RF}')
print(f'ROC AUC: {roc_aucRF}')
print('\n')

# Predecir en el conjunto de prueba
y_pred2 = pipelineDT.predict(X_test2)

# Calcular métricas
accuracyDT = accuracy_score(y_test2, y_pred2)
recallDT = recall_score(y_test2, y_pred2)
f1DT = f1_score(y_test2, y_pred2)
roc_aucDT = roc_auc_score(y_test2, y_pred2)

# Mostrar métricas
print('Decision Tree')
print(f'Accuracy: {accuracyDT}')
print(f'Recall: {recallDT}')
print(f'F1 Score: {f1DT}')
print(f'ROC AUC: {roc_aucDT}')
print('\n')

"""## Visualización de Resultados"""

# Datos
models = ['Logistic Regression', 'Random Forest', 'Decision Tree']
accuracy = [accuracyLR, accuracyRF, accuracyDT]
recall = [recallLR, recallRF, recallDT]
f1 = [f1LR, f1RF, f1DT]
roc_auc = [roc_aucLR, roc_aucRF, roc_aucDT]

# Configurar el ancho de las barras
bar_width = 0.2
index = np.arange(len(models))

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Plotear las barras
rects1 = ax.bar(index, accuracy, bar_width, label='Accuracy')
rects2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
rects3 = ax.bar(index + 2 * bar_width, f1, bar_width, label='F1 Score')
rects4 = ax.bar(index + 3 * bar_width, roc_auc, bar_width, label='ROC AUC')

# Configurar etiquetas y título
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(models)

# Configurar el formato de las etiquetas en el eje y para mayor precisión
ax.yaxis.set_major_formatter('{:.2f}'.format)

# Agregar etiquetas en la parte superior de las barras
for i, rect in enumerate(rects1 + rects2 + rects3 + rects4):
    height = rect.get_height()
    ax.annotate(f'{height:.6f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Mover la leyenda fuera de la gráfica
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Mostrar la gráfica
plt.tight_layout()  # Ajustar el diseño para evitar recorte
plt.show()

# Calcular la matriz de confusión de Logistic Regression
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')

# Calcular la matriz de confusión
cm = confusion_matrix(y_test1, y_pred1)

# Visualizar la matriz de confusión de Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')

# Calcular la matriz de confusión de Decision Tree
cm = confusion_matrix(y_test2, y_pred2)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión de Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')

# Obtener la probabilidad de predicción para la clase positiva
y_prob = pipelineLR.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Visualizar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Logistic Regression')
plt.show()
print('\n')

# Obtener la probabilidad de predicción para la clase positiva
y_prob1 = pipelineRF.predict_proba(X_test1)[:, 1]

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test1, y_prob1)

# Visualizar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Random Forest')
plt.show()
print('\n')

# Obtener la probabilidad de predicción para la clase positiva
y_prob2 = pipelineDT.predict_proba(X_test2)[:, 1]

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test2, y_prob2)

# Visualizar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Decision Tree')
plt.show()
print('\n')

"""## Selección del modelo

El modelo que elegimos, luego de analizar las métricas obtenidas, es Decision Tree. Esta decisión está fundamentada de acuerod a los siguiente:


* Precisión (Accuracy): Es la proporción de predicciones correctas entre el total
de casos. Es una buena medida general, pero puede ser engañosa si las clases están desbalanceadas.

* Recuperación (Recall): Es la proporción de casos positivos reales que fueron identificados correctamente. Es crucial si los falsos negativos son más problemáticos que los falsos positivos.

* Puntuación F1 (F1 Score): Es el promedio armónico de la precisión y la recuperación. Es útil cuando quieres un balance entre precisión y recuperación, especialmente si las clases están desbalanceadas.

* ROC AUC: Mide la capacidad del modelo para distinguir entre clases. Un valor más alto indica una mejor discriminación.


Ahora, evaluando los modelos implementados:

* Regresión Logística: Cuenta el menor valor de precisión, pero el mayor valor de ROC AUC, indicando así una mejor capacidad para distinguir entre clases.

* Bosque Aleatorio y Árbol de Decisión: Extrañamente, estos dos moedlos tienen exactamente las mismas métricas. La precisión de ambos es ligeramente superior a la regresión logística, pero su ROC AUC es marginalmente menor.

Por último, ya que el objetivo principal del proyecto es maximizar la capacidad general de predicción (precisión), el Bosque Aleatorio o el Árbol de Decisión podrían ser ligeramente mejores. Entre estos dos modelos, decidimos utilizar Árbol de Decisión

# Implementación del modelo elegido con un nuevo set de datos

Información del dataset:

Casi 30.000 canciones de la API de Spotify. Consulte el archivo Léame para obtener una tabla de diccionario de datos formateada.

Directorio de datos

| Variable               | Class    | Description                                                                                                       |
|------------------------|----------|-------------------------------------------------------------------------------------------------------------------|
| track_id               | Character| Identificador único de la canción                                                                                |
| track_name             | Character| Name of the song                                                                                                 |
| track_artist           | Character| Artist of the song                                                                                               |
| track_popularity       | Double   | Song popularity (0-100), where higher is better                                                                 |
| track_album_id         | Character| Unique identifier of the album                                                                                   |
| track_album_name       | Character| Name of the album of the song                                                                                    |
| track_album_release_date| Character| Release date of the album                                                                                        |
| playlist_name          | Character| Name of the playlist                                                                                             |
| playlist_id            | Character| Playlist identifier                                                                                             |
| playlist_genre         | Character| Playlist genre                                                                                                   |
| playlist_subgenre      | Character| Playlist subgenre                                                                                                |
| danceability           | Double   | Danceability describes how suitable a track is for dancing. 0.0 is least danceable, 1.0 is most danceable          |
| energy                 | Double   | Energy is a measure from 0.0 to 1.0 representing a perceptual measure of intensity and activity                   |
| key                    | Double   | The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation                 |
| loudness               | Double   | The overall loudness of a track in decibels (dB). Values typically range between -60 and 0 dB                      |
| mode                   | Double   | Mode indicates the modality (major or minor) of a track. Major is represented by 1 and minor is 0                   |
| speechiness            | Double   | La speechiness detecta la presencia de palabras habladas en una pista                                              |
| acousticness           | Double   | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic |
| instrumentalness       | Double   | Predicts whether a track contains no vocals                                                                      |
| liveness               | Double   | Detects the presence of an audience in the recording                                                             |
| valence                | Double   | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track                                |
| tempo                  | Double   | The overall estimated tempo of a track in beats per minute (BPM)                                                  |
| duration_ms            | Double   | Duration of song in milliseconds                                                                                 |

## Importación de librerías
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

"""## EDA"""

df = pd.read_csv('/spotify_songs.csv')
df

"""Información general del dataset"""

df.info()

"""Columnas a evaluar"""

columns = [

    'danceability',
    'energy',
    'key',
    'mode',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]

df[columns].head()

"""Valores únicos por columna"""

for c in df[columns]:
    print(f'columna: {c}, Valores únicos: {df[c].nunique()}')

"""## Transformación de datos y selección de variable objetivo

Normalizar datos
"""

var = [
    'danceability',
    'energy',
    'key',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df[var])
scaled_df[:2]

"""Separación de características y objetivo"""

# Se buscará predecir el modality de una canción, para saber si es mayor o menor


X = scaled_df
y = df['mode']

"""Separar datos para entrenamiento y pruebas"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Entrenamiento del modelo"""

dt_final = DecisionTreeClassifier()
dt_final.fit(X_train, y_train)

"""Métricas"""

# Predecir en el conjunto de prueba
y_pred = dt_final.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Mostrar métricas
print('Decision Tree')
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print('\n')

# Métricas
metrics = ['Accuracy', 'Recall', 'F1 Score', 'ROC AUC']
values = [accuracy, recall, f1, roc_auc]

# Configurar el ancho de las barras
bar_width = 0.5

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(8, 6))

# Plotear las barras
bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])

# Agregar etiquetas en la parte superior de las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

# Configurar el título y las etiquetas
plt.title('Decision Tree Metrics')
plt.ylabel('Metric Values')

# Mostrar la gráfica
plt.show()

"""Matriz de confusión"""

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')

"""Curva ROC"""

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test2, y_prob2)

# Visualizar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Decision Tree')
plt.show()
print('\n')

"""Visualización del árbol"""

plt.figure(figsize=(16,10))
tree.plot_tree(dt_final, feature_names = var,
               class_names = ['Minor', 'Major'], # Major is represented by 1 and minor is 0.
                filled=True)
plt.show()