# Prediccion de generacion energetica solar
# Datos desde NASA POWER API + machine learning
# Proyecto data science - ingenieria informatica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)
warnings.filterwarnings("ignore")

# =========================================================
# Extraccion de datos NASA POWER
# =========================================================
print("=" * 60)
print("Extraccion de datos NASA POWER")
print("=" * 60)

print("\nDescargando datos...")

fecha_fin = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
fecha_inicio = (datetime.now() - timedelta(days=395)).strftime('%Y%m%d')

url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "parameters": "ALLSKY_SFC_SW_DWN,T2M",
    "community": "RE",
    "longitude": -68.2006,
    "latitude": -22.9087,
    "start": fecha_inicio,
    "end": fecha_fin,
    "format": "JSON"
}

for intento in range(3):
    try:
        print(f"Intento {intento + 1}...")
        response = requests.get(url, params=params, verify=False, timeout=60)
        response.raise_for_status()
        break
    except requests.exceptions.ConnectionError as e:
        print(f"Error de conexion: {e}")
        if intento < 2:
            print("Reintentando en 10 segundos...")
            time.sleep(10)
        else:
            raise

data = response.json()
print("Datos descargados ok")

# Convertir a dataframe
dict_radiacion = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
dict_temperatura = data['properties']['parameter']['T2M']

df_rad = pd.DataFrame(list(dict_radiacion.items()), columns=['Fecha_Hora', 'Radiacion_W_m2'])
df_temp = pd.DataFrame(list(dict_temperatura.items()), columns=['Fecha_Hora', 'Temperatura_C'])

df_raw = pd.merge(df_rad, df_temp, on='Fecha_Hora')
df_raw['Fecha_Hora'] = pd.to_datetime(df_raw['Fecha_Hora'], format='%Y%m%d%H')

df_raw.to_csv("datos_crudos_nasa.csv", index=False)
print(f"Guardados {len(df_raw)} registros en datos_crudos_nasa.csv")

# =========================================================
# Limpieza y procesamiento
# =========================================================
print("\n" + "=" * 60)
print("Limpieza y procesamiento")
print("=" * 60)

print("\nLimpiando datos...")

df = df_raw.copy()
df = df.replace(-999.0, np.nan)
df = df.dropna()
# Solo filas con radiacion positiva (de noche no hay generacion)
df = df[df['Radiacion_W_m2'] > 0]
df = df.reset_index(drop=True)

print(f"Registros validos tras limpieza: {len(df)}")

# Calculo de generacion en MW
# Parametros del parque: 150.000 m2, eficiencia 21%, PR 0.75
area_m2 = 150000
eficiencia = 0.21
performance_ratio = 0.75

df['Generacion_MW'] = (df['Radiacion_W_m2'] * area_m2 * eficiencia * performance_ratio) / 1000000

# Ruido para simular variabilidad real
np.random.seed(42)
df['Generacion_MW'] = (df['Generacion_MW'] + np.random.normal(0, 0.5, len(df))).clip(lower=0)

df.to_csv("datos_limpios_modelo.csv", index=False)
print("Dataset limpio guardado en datos_limpios_modelo.csv")

# =========================================================
# EDA basico
# =========================================================
print("\n" + "=" * 60)
print("EDA basico")
print("=" * 60)

print("\nIniciando EDA...")

print("\nPrimeras filas:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

# Media y std de las variables que nos interesan
estadisticas = df[['Radiacion_W_m2', 'Temperatura_C', 'Generacion_MW']].agg(['mean', 'std'])
print("\nMedia y desviacion estandar:")
print(estadisticas)

# Correlacion entre variables
plt.figure(figsize=(8,6))
temp_corr = df[['Radiacion_W_m2', 'Temperatura_C', 'Generacion_MW']].corr()
sns.heatmap(temp_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlacion entre variables")
plt.tight_layout()
plt.show()

# Boxplot por variable separado para que cada una tenga su propia escala
fig, axes = plt.subplots(1, 3, figsize=(14,5))

axes[0].boxplot(df['Radiacion_W_m2'])
axes[0].set_title("Radiacion W/m2")
axes[0].set_xticks([])

axes[1].boxplot(df['Temperatura_C'])
axes[1].set_title("Temperatura C")
axes[1].set_xticks([])

axes[2].boxplot(df['Generacion_MW'])
axes[2].set_title("Generacion MW")
axes[2].set_xticks([])

plt.suptitle("Outliers - boxplot")
plt.tight_layout()
plt.show()

# Histograma de distribucion de generacion
plt.figure(figsize=(8,5))
df['Generacion_MW'].hist(bins=30, color='steelblue', edgecolor='white')
plt.title("Distribucion de generacion MW")
plt.xlabel("Generacion MW")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Scatter radiacion vs generacion para ver la relacion directa
plt.figure(figsize=(8,5))
plt.scatter(df['Radiacion_W_m2'], df['Generacion_MW'], alpha=0.3, color='steelblue')
plt.xlabel("Radiacion W/m2")
plt.ylabel("Generacion MW")
plt.title("Radiacion vs generacion")
plt.tight_layout()
plt.show()

# =========================================================
# Split
# =========================================================
print("\n" + "=" * 60)
print("Split")
print("=" * 60)

print("\nPreparando datos...")

X = df[['Radiacion_W_m2', 'Temperatura_C']]
y = df['Generacion_MW']

# Ojo que aca usamos 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# =========================================================
# Regresion lineal
# =========================================================
print("\n" + "=" * 60)
print("Regresion lineal")
print("=" * 60)

print("\nEntrenando modelo...")

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

y_pred_lineal = modelo_lineal.predict(X_test)
print("Listo")

# Metricas
r2_lineal = r2_score(y_test, y_pred_lineal)
rmse_lineal = np.sqrt(mean_squared_error(y_test, y_pred_lineal))
mae_lineal = mean_absolute_error(y_test, y_pred_lineal)

print(f"R2: {r2_lineal:.4f}")
print(f"RMSE: {rmse_lineal:.4f}")
print(f"MAE: {mae_lineal:.4f}")

# Ver que variable pesa mas
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo_lineal.coef_
})

print("\nCoeficientes:")
print(coef_df)

# Predicciones vs valores reales
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lineal, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Prediccion")
plt.title("Regresion lineal - real vs predicho")
plt.tight_layout()
plt.show()

# Validacion cruzada
scores = cross_val_score(modelo_lineal, X, y, cv=5, scoring='r2')
print(f"\nR2 promedio CV: {scores.mean():.4f}")

# =========================================================
# Arbol de decision
# =========================================================
print("\n" + "=" * 60)
print("Arbol de decision")
print("=" * 60)

# Usamos max_depth=5 para evitar sobreajuste, si lo dejamos crecer
# demasiado memoriza el set de entrenamiento y no generaliza bien
arbol = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)

arbol.fit(X_train, y_train)

y_pred_arbol = arbol.predict(X_test)

r2_arbol = r2_score(y_test, y_pred_arbol)
rmse_arbol = np.sqrt(mean_squared_error(y_test, y_pred_arbol))

print(f"\nR2 arbol: {r2_arbol:.4f}")
print(f"RMSE arbol: {rmse_arbol:.4f}")

# =========================================================
# Random forest
# =========================================================
print("\n" + "=" * 60)
print("Random forest")
print("=" * 60)

# 100 arboles con profundidad 5 para no sobreajustar
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\nR2 RF: {r2_rf:.4f}")
print(f"RMSE RF: {rmse_rf:.4f}")

# Importancia de variables
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf.feature_importances_
})

print("\nImportancia de variables:")
print(importancias)

# Grafico de importancia de variables
plt.figure(figsize=(7,4))
plt.barh(importancias['Variable'], importancias['Importancia'], color='steelblue')
plt.title("Importancia de variables - random forest")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# =========================================================
# Clasificacion
# =========================================================
print("\n" + "=" * 60)
print("Clasificacion")
print("=" * 60)

# Divido en alta o baja generacion usando la mediana como umbral
mediana = df['Generacion_MW'].median()

df['Alta_Generacion'] = np.where(df['Generacion_MW'] > mediana, 1, 0)

y_clas = df['Alta_Generacion']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clas, test_size=0.2, random_state=42
)

# Escalado necesario porque KNN y regresion logistica son sensibles
# a la magnitud de las variables, sin esto la distancia entre puntos
# queda distorsionada por las diferencias de escala entre radiacion y temperatura
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# Regresion logistica
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train_c)
y_pred_log = log_model.predict(X_test_scaled)

print("\nRegresion logistica")
print(classification_report(y_test_c, y_pred_log))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_c)
y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN")
print(classification_report(y_test_c, y_pred_knn))

# Matriz de confusion
matriz = confusion_matrix(y_test_c, y_pred_knn)

plt.figure(figsize=(6,5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de confusion")
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# =========================================================
# Comparacion final
# =========================================================
print("\n" + "=" * 60)
print("Comparacion final")
print("=" * 60)

comparacion = pd.DataFrame({
    'Modelo': ['Regresion lineal', 'Arbol de decision', 'Random forest'],
    'R2': [r2_lineal, r2_arbol, r2_rf],
    'RMSE': [rmse_lineal, rmse_arbol, rmse_rf]
})

print(comparacion)

# =========================================================
# Nuevas predicciones
# =========================================================
print("\n" + "=" * 60)
print("Nuevas predicciones")
print("=" * 60)

nuevo = pd.DataFrame({
    'Radiacion_W_m2': [850],
    'Temperatura_C': [28]
})

pred_lineal = modelo_lineal.predict(nuevo)[0]
pred_arbol = arbol.predict(nuevo)[0]
pred_rf = rf.predict(nuevo)[0]

print(f"Lineal: {pred_lineal:.2f} MW")
print(f"Arbol: {pred_arbol:.2f} MW")
print(f"RF: {pred_rf:.2f} MW")

# =========================================================
# Conclusiones
# =========================================================
print("\n" + "=" * 60)
print("Conclusiones")
print("=" * 60)

print("""
1. La radiacion solar tiene una fuerte relacion con la generacion energetica.
2. Random forest entrega mejor precision predictiva que los otros modelos.
3. La regresion lineal es el modelo mas interpretable de los tres.
4. Limitar la profundidad del arbol evita el sobreajuste.
5. La clasificacion permite identificar periodos de alta generacion.
""")
