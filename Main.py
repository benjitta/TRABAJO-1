

#parte benja valverde (eliminar esto depues de colocar la parte de arriba)

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
