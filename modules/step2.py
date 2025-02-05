
import streamlit as st
import numpy as np
from modules.utils import detect_variable_type, reset_steps

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
        dataset = dataset.replace(['None', '', 'nan', 'Nan', 'NaN'], np.nan)
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
