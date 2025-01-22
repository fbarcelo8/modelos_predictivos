
import streamlit as st
import pandas as pd

def step_9():
    if not st.session_state.get('step_9_enabled', False):
        return

    st.header("Paso 9: Visualización de Resultados")
    model = st.session_state['trained_model']
    dataset = st.session_state['data']
    fixed_predictors = st.session_state['fixed_predictors']
    candidate_predictors = st.session_state['candidate_predictors']
    target = st.session_state['target']

    predictors = fixed_predictors + candidate_predictors
    X = dataset[predictors]
    y = dataset[target]

    test_size = 1 - st.session_state.get('train_size', 0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)

    # Comparación de valores reales vs predichos
    comparison = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    st.write("**Comparación entre valores reales y predichos:**")
    st.dataframe(comparison)

    # Descargar resultados
    csv_data = comparison.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Resultados",
        data=csv_data,
        file_name="resultados_modelo.csv",
        mime="text/csv"
    )
  
