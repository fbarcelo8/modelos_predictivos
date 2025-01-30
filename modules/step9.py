
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from modules.utils import reset_steps

def step_9():
    if not st.session_state.get('step_9_enabled', False):
        return

    st.header("Paso 9: Visualización de Resultados")
    model = st.session_state['trained_model']
    dataset = st.session_state['data']
    target = st.session_state['target']

    predictors = st.session_state.get('selected_features', [])
    X = dataset[predictors]
    y = dataset[target]

    test_size = 1 - st.session_state.get('train_size', 0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)

    # Comparación de valores reales vs predichos
    # Convertir a enteros para evitar checkmarks en la tabla
    comparison = pd.DataFrame({
        "Real": y_test.astype(int).values,  # Convertir valores categóricos a enteros
        "Predicción": y_pred.astype(int)  # Convertir predicciones a enteros
    })
    st.write("**Comparación entre valores reales y predichos:**")
    st.dataframe(comparison)

    # Sección 9.1: Descargar resultados
    st.subheader("Paso 9.1: Descargar Resultados")

    # Descargar resultados como CSV
    csv_data = comparison.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📂 Descargar Resultados (CSV)",
        data=csv_data,
        file_name="resultados_modelo.csv",
        mime="text/csv"
    )

    # Guardar el modelo en formato .model usando pickle
    model_filename = "automodeler_model.model"
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    # Descargar el modelo entrenado en formato .model
    with open(model_filename, "rb") as model_file:
        st.download_button(
            label="📥 Descargar Modelo (.model)",
            data=model_file,
            file_name="automodeler_model.model",
            mime="application/octet-stream"
        )
