# =========================================================
# PROYECTO DATA SCIENCE
# Predicción de Generación Energética Solar
# API NASA POWER + Machine Learning
# =========================================================

# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings

from datetime import datetime, timedelta

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================================================
# FASE 1 — EXTRACCIÓN DE DATOS NASA POWER
# =========================================================

print("=" * 60)
print("FASE 1 — EXTRACCIÓN DE DATOS NASA POWER")
print("=" * 60)

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

response = requests.get(
    url,
    params=params,
    verify=False,
    timeout=30
)

response.raise_for_status()

data = response.json()

print("\nDatos descargados correctamente desde NASA POWER.")

# =========================================================
# CONVERSIÓN A DATAFRAME
# =========================================================

dict_radiacion = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']

dict_temperatura = data['properties']['parameter']['T2M']

df_rad = pd.DataFrame(
    list(dict_radiacion.items()),
    columns=['Fecha_Hora', 'Radiacion_W_m2']
)

df_temp = pd.DataFrame(
    list(dict_temperatura.items()),
    columns=['Fecha_Hora', 'Temperatura_C']
)

df_raw = pd.merge(
    df_rad,
    df_temp,
    on='Fecha_Hora'
)

df_raw['Fecha_Hora'] = pd.to_datetime(
    df_raw['Fecha_Hora'],
    format='%Y%m%d%H'
)

# Exportar datos crudos
df_raw.to_csv("datos_crudos_nasa.csv", index=False)

print(f"\nDatos crudos guardados ({len(df_raw)} registros).")

# =========================================================
# FASE 2 — LIMPIEZA Y PROCESAMIENTO
# =========================================================

print("\n" + "=" * 60)
print("FASE 2 — LIMPIEZA Y PROCESAMIENTO")
print("=" * 60)

df = df_raw.copy()

# Reemplazar valores inválidos
df = df.replace(-999.0, np.nan)

# Eliminar nulos
df = df.dropna()

# Mantener solo horas con radiación positiva
df = df[df['Radiacion_W_m2'] > 0]

df = df.reset_index(drop=True)

print(f"\nRegistros válidos: {len(df)}")

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\nCalculando generación energética...")

area_m2 = 150000

eficiencia = 0.21

performance_ratio = 0.75

df['Generacion_MW'] = (
    df['Radiacion_W_m2']
    * area_m2
    * eficiencia
    * performance_ratio
) / 1000000

# Agregar ruido para simulación realista
np.random.seed(42)

df['Generacion_MW'] = (
    df['Generacion_MW']
    + np.random.normal(0, 0.5, len(df))
).clip(lower=0)

# Exportar dataset limpio
df.to_csv("datos_limpios_modelo.csv", index=False)

print("\nDataset limpio guardado correctamente.")

#EDA básico

print("\nIniciando EDA...")

print("\nPrimeras filas:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

# media y std de las variables que nos interesan
estadisticas = df[['Radiacion_W_m2', 'Temperatura_C', 'Generacion_MW']].agg(['mean', 'std'])
print("\nmedia y desviacion estandar:")
print(estadisticas)

# correlacion entre variables
plt.figure(figsize=(8,6))
temp_corr = df[['Radiacion_W_m2', 'Temperatura_C', 'Generacion_MW']].corr()
sns.heatmap(temp_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("correlacion entre variables")
plt.tight_layout()
plt.show()

# boxplot outliers
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Radiacion_W_m2', 'Temperatura_C', 'Generacion_MW']])
plt.title("outliers - boxplot")
plt.tight_layout()
plt.show()


#Split

print("\npreparando datos...")

X = df[['Radiacion_W_m2', 'Temperatura_C']]
y = df['Generacion_MW']

# ojo que acá usamos 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")


#Regresión Lineal

print("\nRegresión lineal:")

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

y_pred_lineal = modelo_lineal.predict(X_test)
print("listo")

# métricas
r2_lineal = r2_score(y_test, y_pred_lineal)
rmse_lineal = np.sqrt(mean_squared_error(y_test, y_pred_lineal))
mae_lineal = mean_absolute_error(y_test, y_pred_lineal)

print(f"R2   : {r2_lineal:.4f}")
print(f"RMSE : {rmse_lineal:.4f}")
print(f"MAE  : {mae_lineal:.4f}")

# ver qué variable pesa más
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo_lineal.coef_
})

print("\ncoeficientes:")
print(coef_df)

#VALIDACIÓN CRUZADA

scores = cross_val_score(
    modelo_lineal,
    X,
    y,
    cv=5,
    scoring='r2'
)

print(f"\nR² promedio CV: {scores.mean():.4f}")

#ÁRBOL DE DECISIÓN

print("\n" + "=" * 60)
print("FASE 6 — ÁRBOL DE DECISIÓN")
print("=" * 60)

arbol = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

arbol.fit(X_train, y_train)

y_pred_arbol = arbol.predict(X_test)

r2_arbol = r2_score(y_test, y_pred_arbol)

rmse_arbol = np.sqrt(
    mean_squared_error(y_test, y_pred_arbol)
)

print(f"\nR² Árbol : {r2_arbol:.4f}")
print(f"RMSE Árbol: {rmse_arbol:.4f}")


#RANDOM FOREST

print("\n" + "=" * 60)
print("FASE 7 — RANDOM FOREST")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)

rmse_rf = np.sqrt(
    mean_squared_error(y_test, y_pred_rf)
)

print(f"\nR² RF   : {r2_rf:.4f}")
print(f"RMSE RF : {rmse_rf:.4f}")

#Importancia variables
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf.feature_importances_
})

print("\nImportancia de Variables:")
print(importancias)

# CLASIFICACIÓN

print("\n" + "=" * 60)
print("FASE 8 — CLASIFICACIÓN")
print("=" * 60)

mediana = df['Generacion_MW'].median()

df['Alta_Generacion'] = np.where(
    df['Generacion_MW'] > mediana,
    1,
    0
)

y_clas = df['Alta_Generacion']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y_clas,
    test_size=0.2,
    random_state=42
)

#Escalado
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_c)

X_test_scaled = scaler.transform(X_test_c)
