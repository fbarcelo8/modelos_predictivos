
import streamlit as st
import pandas as pd
from modules.preprocessing import preprocess_dataset
from modules.utils import reset_steps

def step_1():
    st.header("Paso 1: Subida de Archivo")
    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        if 'uploaded_file_name' in st.session_state and st.session_state['uploaded_file_name'] != file_name:
            st.session_state.clear()
        st.session_state['uploaded_file_name'] = file_name
        
        file_extension = file_name.split(".")[-1]
        try:
            if file_extension == "csv":
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return
        
        # Validar que el dataset no esté vacío
        if dataset.empty:
            st.error("El archivo cargado no contiene datos válidos.")
            return
        
        # Procesar el dataset
        try:
            dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        except Exception as e:
            st.error(f"Error al procesar el dataset: {e}")
            return
        
        # Validar si el dataset sigue siendo válido después del procesamiento
        if dataset.empty:
            st.error("El dataset quedó vacío después del preprocesamiento.")
            return
        
        # Convertir columnas problemáticas
        for col in dataset.columns:
            if dataset[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                dataset[col] = dataset[col].astype(str)
        
        # Guardar dataset en el estado de sesión
        st.session_state['data'] = dataset
        
        # Mostrar tabla con manejo de errores
        try:
            st.write("Vista previa del dataset:")
            st.dataframe(dataset)
        except Exception as e:
            st.error(f"Error al renderizar la tabla: {e}")
            st.write("Dimensiones del dataset:", dataset.shape)
            st.write("Tipos de datos:", dataset.dtypes)
        
        st.session_state['step_2_enabled'] = True

