# AutoModeler

La herramienta **AutoModeler** es una aplicación desarrollada en **Streamlit** que permite la construcción semiautomática de modelos predictivos. Sus principales funcionalidades incluyen:


## 1. Carga y preprocesamiento de datos:
- Permite subir archivos **CSV** o **Excel**.
- Elimina duplicados y columnas con un alto porcentaje de valores faltantes.
- Detecta automáticamente los tipos de variables (numéricas, categóricas binarias y no binarias).
- Permite modificar manualmente los tipos de variables.

## 2. Selección de la variable objetivo (**target**):
- Permite elegir la variable a predecir.
- Elimina registros con valores faltantes en la variable objetivo.
- Si la variable es categórica binaria, permite definir la etiqueta positiva.

## 3. Análisis exploratorio:
- Muestra la distribución de la variable objetivo con tablas y gráficos.
- Evalúa columnas identificadoras y posibles identificadoras para evitar que entren en el modelo.

## 4. Selección del modelo:
- Para variables numéricas: **Regresión Lineal**.
- Para variables categóricas: **Regresión Logística**.
- Define el porcentaje de datos de entrenamiento y prueba.

## 5. Selección de variables **predictoras**:
- Permite seleccionar variables fijas y candidatas.
- Manejo de valores faltantes (eliminación de registros con valores nulos).
- Aplica selección forward **BIC** para encontrar la mejor combinación de variables **predictoras** para el modelo en caso de usar la Regresión Lineal y aplica el criterio de la mejor curva ROC para la regresión logística.

## 6. Entrenamiento del modelo:
- Aplica preprocesamiento con **StandardScaler** y codificación de variables categóricas (**One-Hot Encoding**).
- Utiliza **Regresión Lineal** o **Regresión Logística** según el tipo de problema.
- Muestra los coeficientes y **p-valores** del modelo ajustado.

## 7. Evaluación del modelo:
- Para regresión lineal: **MSE**, **MAE** y **R²**.
- Para regresión logística: **Accuracy**, **Precision**, **Specificity**, **Recall**, **F1-Score**, **AUC**, **Matriz de Confusión**.
- Gráficos de **curva ROC** para modelos de clasificación.

## 8. Visualización y descarga de resultados:
- Comparación entre valores reales y predichos.
- Opción para descargar los resultados en **CSV**.
- Permite guardar y descargar el modelo entrenado en formato **.model**.
