
import streamlit as st
import pandas as pd
from modules.preprocessing import preprocess_dataset, normalize_dataframe
from modules.utils import reset_steps

def step_1():
    """
    Función para la subida y visualización del archivo. Permite cargar datasets en formato
    CSV o Excel, realiza el preprocesamiento y muestra los resultados en Streamlit.
    """

    st.header("Paso 1: Subida de Archivo")
    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        if 'uploaded_file_name' in st.session_state and st.session_state['uploaded_file_name'] != file_name:
            st.session_state.clear()
            st.session_state['uploaded_file_name'] = file_name
        elif 'uploaded_file_name' not in st.session_state:
            st.session_state['uploaded_file_name'] = file_name

        # Detectar la extensión del archivo y cargarlo
        file_extension = file_name.split(".")[-1]
        try:
            if file_extension == "csv":
                dataset = pd.read_csv(uploaded_file, encoding="utf-8")
            elif file_extension in ["xlsx", "xls"]:
                dataset = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.error("Formato de archivo no soportado.")
                return

            # Preprocesar el dataset
            dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)

            # Normalizar los tipos de datos
            dataset = normalize_dataframe(dataset)

            # Guardar el dataset en el estado de la sesión
            st.session_state['data'] = dataset

            # Mostrar resultados del preprocesamiento
            st.success("¡Dataset cargado y preprocesado exitosamente!")
            st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
            if dropped_columns:
                st.markdown(
                    f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**."
                )
            else:
                st.markdown("No se eliminaron columnas por valores faltantes.")
            
            # Verificar el DataFrame antes de mostrarlo
            st.write("Tipos de columnas en el dataset:")
            st.write(dataset.dtypes)  # Mostrar los tipos de datos
            st.write("Vista previa del dataset limpio:")
            st.write(dataset.head(10))  # Mostrar solo las primeras 10 filas

            # Habilitar paso 2
            st.session_state['step_2_enabled'] = True

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
