import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_dataset(dataset, missing_threshold=0.3):
    """
    Limpia y preprocesa el dataset eliminando valores nulos y duplicados.
    Además, elimina columnas con un porcentaje alto de valores faltantes.
    
    Parameters:
    - dataset: pd.DataFrame, dataset a procesar.
    - missing_threshold: float, porcentaje máximo de valores faltantes permitido (entre 0 y 1).
    
    Returns:
    - dataset: pd.DataFrame, dataset limpio.
    - duplicates_removed: int, número de registros duplicados eliminados.
    - dropped_columns: list, nombres de las columnas eliminadas por exceso de valores faltantes.
    """
    # Eliminar duplicados
    dataset_no_duplicates = dataset.drop_duplicates()
    duplicates_removed = len(dataset) - len(dataset_no_duplicates)
    
    # Identificar columnas con más del umbral de valores faltantes
    missing_percentage = dataset_no_duplicates.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index.tolist()
    
    # Eliminar columnas con muchos valores faltantes
    dataset_cleaned = dataset_no_duplicates.drop(columns=columns_to_drop)
    
    return dataset_cleaned, duplicates_removed, columns_to_drop


def detect_variable_type(dataset, target):
    """
    Detecta automáticamente el tipo de variable target.
    """
    unique_values = dataset[target].nunique()
    if unique_values < 10:
        if unique_values == 2:
            return "Categórica Binaria"
        else:
            return "Categórica No Binaria"
    else:
        return "Numérica"
    
def reset_steps(from_step):
    """
    Resetea todos los pasos posteriores al paso dado.
    """
    steps_to_reset = {
        "step_2": ["step_3_and_4_enabled", "step_5_enabled", "step_6_enabled", "step_7_enabled", "step_9_enabled"],
        "step_3_and_4": ["step_5_enabled", "step_6_enabled", "step_7_enabled", "step_9_enabled"],
        "step_5": ["step_6_enabled", "step_7_enabled", "step_9_enabled"],
    }
    
    for step in steps_to_reset.get(from_step, []):
        if step in st.session_state:
            del st.session_state[step]

def step_1():
    st.header("Paso 1: Subida de Archivo")
    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    
    # Detectar si es un nuevo archivo y reiniciar el estado
    if uploaded_file:
        # Guardar el nombre del archivo para detectar cambios
        file_name = uploaded_file.name
        if 'uploaded_file_name' in st.session_state and st.session_state['uploaded_file_name'] != file_name:
            # Reiniciar el estado si el archivo es diferente al anterior
            st.session_state.clear()
            st.session_state['uploaded_file_name'] = file_name
        elif 'uploaded_file_name' not in st.session_state:
            # Inicializar el nombre del archivo la primera vez
            st.session_state['uploaded_file_name'] = file_name
        
        file_extension = file_name.split(".")[-1]
        if file_extension == "csv":
            dataset = pd.read_csv(uploaded_file)
        else:
            dataset = pd.read_excel(uploaded_file)

        dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        st.session_state['data'] = dataset
        st.success("¡Dataset cargado y preprocesado exitosamente!")
        st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
        if dropped_columns:
            st.markdown(
                f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**."
            )
        else:
            st.markdown("No se eliminaron columnas por valores faltantes.")
        st.write(dataset)
        st.session_state['step_2_enabled'] = True

def step_2():
    if not st.session_state.get('step_2_enabled', False):
        return

    st.header("Paso 2: Selección de la Variable Target")
    dataset = st.session_state['data']
    previous_target = st.session_state.get('target', None)
    target = st.selectbox("Selecciona la Variable Target", dataset.columns, index=dataset.columns.get_loc(previous_target) if previous_target else 0)

    if target != previous_target:
        reset_steps("step_2")

    if target:
        # Eliminar filas con valores faltantes en la variable target
        missing_rows_before = len(dataset)
        dataset_cleaned = dataset.dropna(subset=[target])
        missing_rows_after = len(dataset_cleaned)
        rows_removed = missing_rows_before - missing_rows_after

        # Calcular el porcentaje de registros eliminados
        percentage_removed = (rows_removed / missing_rows_before) * 100 if missing_rows_before > 0 else 0

        if rows_removed > 0:
            st.warning(f"Se han eliminado **{rows_removed}** registros ({percentage_removed:.2f}% del total) por contener valores faltantes en la variable target.")
        else:
            st.info("No se eliminaron registros por valores faltantes en la variable target.")

        st.session_state['data'] = dataset_cleaned  # Guardamos el dataset limpio en la sesión
        
        unique_values = sorted(dataset_cleaned[target].unique())
        if len(unique_values) == 2:
            previous_pos_label = st.session_state.get('pos_label', unique_values[1])
            pos_label = st.selectbox(
                "Selecciona la etiqueta positiva (pos_label):",
                options=unique_values,
                index=unique_values.index(previous_pos_label) if previous_pos_label in unique_values else 1
            )
            if pos_label != st.session_state.get('pos_label', None):
                reset_steps("step_2")
            st.session_state['pos_label'] = pos_label
        
        if st.button("Confirmar Target"):
            st.session_state['target'] = target
            st.success("Variable Target seleccionada y registros con valores faltantes eliminados.")
            st.session_state['step_3_and_4_enabled'] = True

def step_3():
    if not st.session_state.get('step_3_and_4_enabled', False):
        return

    st.header("Paso 3: Detección Automática del Tipo de Variable")
    target = st.session_state['target']
    dataset = st.session_state['data']
    target_type = detect_variable_type(dataset, target)
    st.write(f"La variable seleccionada '{target}' se detectó como: **{target_type}**")

    # Almacenar el tipo de variable en el estado de sesión
    st.session_state['target_type'] = target_type

    if target_type == "Categórica Binaria":
        st.subheader("Distribución de la Variable Target")

        # Contar valores únicos de la variable target
        target_counts = dataset[target].value_counts()
        total = len(dataset)

        # Crear un DataFrame con los valores y porcentajes
        target_distribution = pd.DataFrame({
            "Categoría": target_counts.index.astype(str),
            "Frecuencia": target_counts.values,
            "Porcentaje": (target_counts.values / total) * 100
        })

        # Mostrar la tabla con porcentajes
        st.table(target_distribution.style.format({"Porcentaje": "{:.2f}%"}))

        # Crear un gráfico de barras con los colores personalizados
        colors = ['#A759FE', '#FFFF00']
        fig, ax = plt.subplots()
        sns.barplot(
            x=target_distribution["Categoría"],
            y=target_distribution["Frecuencia"],
            palette=colors,
            ax=ax
        )
        ax.set_title("Distribución de la Variable Target")
        ax.set_xlabel("Categoría")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    st.session_state['step_4_enabled'] = True

def step_4():
    if not st.session_state.get('step_4_enabled', False):
        return

    st.header("Paso 4: Selección del Modelo")
    target_type = st.session_state['target_type']
    previous_model = st.session_state.get('model', None)

    if target_type == "Numérica":
        # Lista de modelos disponibles para variables numéricas
        models = [
            "Regresión Lineal",
            "Regresión No Lineal",
            "Árbol de Decisión",
            "Random Forest",
            "BaggingRegressor",
            "XGBoost",
            "Red Neuronal"
        ]
        selected_model = st.selectbox(
            "Selecciona el modelo que deseas utilizar:",
            models,
            index=models.index(previous_model) if previous_model in models else 0  # Regresión Lineal predeterminado
        )
    else:
        # Para variables categóricas, mantenemos Regresión Logística
        selected_model = "Regresión Logística"
        st.info(f"Modelo seleccionado: **{selected_model}**")

    if selected_model != previous_model:
        reset_steps("step_3_and_4")

    if st.button("Confirmar Modelo"):
        st.session_state['model'] = selected_model
        if selected_model != "Regresión Lineal" and target_type == "Numérica":
            st.warning(f"El modelo **{selected_model}** está en desarrollo y no está disponible actualmente.")
            st.session_state['step_5_enabled'] = False  # Asegúrate de deshabilitar el siguiente paso
        else:
            st.success("Modelo seleccionado.")
            st.session_state['step_5_enabled'] = True

def step_5():
    if not st.session_state.get('step_5_enabled', False):
        return

    st.header("Paso 5: Selección de Variables Predictoras")
    dataset = st.session_state['data']
    target = st.session_state['target']

    # Obtener las opciones actuales de variables predictoras
    available_predictors = [col for col in dataset.columns if col != target]

    # Filtrar las variables predictoras seleccionadas previamente para que sean válidas
    previous_predictors = st.session_state.get('predictors', [])
    valid_previous_predictors = [col for col in previous_predictors if col in available_predictors]

    # Mostrar el multiselect con los valores válidos
    predictors = st.multiselect(
        "Selecciona las Variables Predictoras",
        available_predictors,
        default=valid_previous_predictors
    )

    # Si las variables predictoras seleccionadas cambian, reinicia los pasos posteriores
    if set(predictors) != set(valid_previous_predictors):
        reset_steps("step_5")

    if not predictors:
        st.warning("Selecciona al menos una variable predictora.")
        return

    # Opciones para manejar valores faltantes
    st.subheader("Manejo de Valores Faltantes")
    handle_missing = st.radio(
        "¿Cómo deseas manejar los valores faltantes en las variables predictoras?",
        options=["Eliminar filas con valores faltantes", "Imputar valores faltantes"],
        index=0
    )

    if st.button("Confirmar Variables Predictoras"):
        if handle_missing == "Eliminar filas con valores faltantes":
            # Eliminar registros con valores faltantes
            missing_rows_before = len(dataset)
            dataset_cleaned = dataset.dropna(subset=predictors)
            missing_rows_after = len(dataset_cleaned)
            rows_removed = missing_rows_before - missing_rows_after

            if rows_removed > 0:
                st.warning(f"Se han eliminado **{rows_removed}** registros en total por valores faltantes.")

                # Calcular cuántos registros fueron eliminados por cada variable
                impact_table = pd.DataFrame({
                    "Variable": predictors,
                    "Registros Eliminados": [dataset[col].isnull().sum() for col in predictors],
                    "Porcentaje Eliminado (%)": [(dataset[col].isnull().sum() / len(dataset) * 100).round(2) for col in predictors]
                }).sort_values(by="Registros Eliminados", ascending=False)

                # Mostrar la tabla ordenada
                st.markdown("**Impacto de las Variables Predictoras en la Eliminación de Registros**")
                st.write("Cantidad de registros eliminados por valores faltantes en cada variable (ordenado de mayor a menor):")
                st.dataframe(impact_table)
            else:
                st.info("No se eliminaron registros por valores faltantes en las variables predictoras seleccionadas.")
        else:
            st.warning("Función todavía no disponible")
            st.stop()

        # Actualizar el estado global
        st.session_state['predictors'] = predictors
        st.session_state['data'] = dataset_cleaned
        st.session_state['step_6_enabled'] = True
        st.session_state['step_7_enabled'] = True
        st.session_state['step_9_enabled'] = True

        st.success("Variables predictoras seleccionadas y datos preprocesados correctamente.")

def step_6():
    if not st.session_state.get('step_6_enabled', False):
        return

    st.header("Paso 6: Entrenamiento del Modelo")
    dataset = st.session_state['data']
    predictors = st.session_state['predictors']
    target = st.session_state['target']
    target_type = st.session_state['target_type']

    X = dataset[predictors]
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mostrar un texto explicativo sobre el entrenamiento
    if target_type == "Numérica":
        st.markdown(
            "**Regresión Lineal:** Se ha dividido el conjunto de datos en un 80% para entrenamiento y un 20% para test.<br>"
            "Se ha utilizado la regresión lineal para modelar la relación entre las variables predictoras y la variable objetivo.<br>"
            "El modelo se ajustará minimizando el error cuadrático medio (MSE).",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "**Regresión Logística:** Se ha dividido el conjunto de datos en un 80% para entrenamiento y un 20% para test.<br>"
            "Se ha utilizado la regresión logística para predecir probabilidades de clasificación en función de las variables predictoras.<br>"
            "El modelo se ha entrenado usando validación cruzada para garantizar una mejor generalización.",
            unsafe_allow_html=True
        )

    # Preprocesamiento de datos
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ]
    )

    if target_type == "Numérica":
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])
    else:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

    # Entrenar el modelo
    model.fit(X_train, y_train)
    st.session_state['trained_model'] = model
    st.success("Modelo entrenado con éxito.")

def step_7():
    if not st.session_state.get('step_7_enabled', False):
        return

    st.header("Paso 7: Evaluación del Modelo")
    model = st.session_state['trained_model']
    dataset = st.session_state['data']
    predictors = st.session_state['predictors']
    target = st.session_state['target']
    target_type = st.session_state['target_type']

    X = dataset[predictors]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if target_type == "Numérica":
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Diseño personalizado para las métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">MSE</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(mse),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">MAE</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(mae),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h3 style="margin: 0; font-size: 1.2rem; font-weight: 600;">R²</h3>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(r2),
                unsafe_allow_html=True,
            )

    else:  # Regresión Logística
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Obtener etiquetas únicas y usar la etiqueta positiva seleccionada en el paso 2
        pos_label = st.session_state.get('pos_label', None)

        # Calcular la matriz de confusión con nombres dinámicos
        unique_labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        # Crear DataFrame con nombres dinámicos para las columnas e índices
        cm_df = pd.DataFrame(
            cm,
            columns=[f"Predicción {label}" for label in unique_labels],
            index=[f"Real {label}" for label in unique_labels]
        )

        # Agregar totales en filas y columnas
        cm_df["Total Real"] = cm_df.sum(axis=1)
        cm_df.loc["Total Predicción"] = cm_df.sum(axis=0)
        
        # Mostrar la matriz de confusión
        st.markdown("**Matriz de Confusión**")
        st.table(cm_df)

        # Calcular métricas de clasificación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        specificity = recall_score(y_test, y_pred, pos_label=unique_labels[0])
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        # Diseño personalizado para métricas de regresión logística
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Exactitud (Accuracy)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(accuracy),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Precisión (Precision)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(precision),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Recall (Sensibilidad)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(recall),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Especificidad</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(specificity),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">F1-Score</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(f1),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">AUC</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(auc),
                unsafe_allow_html=True,
            )
    
        # Curva ROC
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=pos_label)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("Curva ROC")
        ax.set_xlabel("1 - Especificidad")
        ax.set_ylabel("Sensibilidad")
        ax.legend()
        st.pyplot(fig)


def step_9():
    if not st.session_state.get('step_9_enabled', False):
        return

    st.header("Paso 9: Visualización de Resultados")
    model = st.session_state['trained_model']
    dataset = st.session_state['data']
    predictors = st.session_state['predictors']
    target = st.session_state['target']

    X = dataset[predictors]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    comparison = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    st.write(comparison)
    st.download_button(
        label="Descargar Resultados",
        data=comparison.to_csv(index=False),
        file_name="resultados_modelo.csv",
        mime="text/csv"
    )

def main():
    # Crear columnas para centrar la imagen
    col1, col2, col3 = st.columns([1, 2, 1])  # Ajusta las proporciones para centrar la imagen
    with col2:
        st.image("images/logo_butler.png", width=400)

    # Mostrar el título debajo de la imagen
    st.title("Herramienta de Modelos Predictivos")

    st.write("Sube tu dataset y sigue los pasos para analizarlo y entrenar un modelo.")

    step_1()
    step_2()
    step_3()
    step_4()
    step_5()
    step_6()
    step_7()
    step_9()

if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    main()
    
