
import streamlit as st
import pandas as pd
from modules.preprocessing import preprocess_dataset
from modules.utils import reset_steps

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

        # Cargar el archivo y manejar errores
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
            st.error("El archivo cargado está vacío. Por favor, sube un archivo válido.")
            return

        # Convertir columnas categóricas a tipo 'object'
        category_columns = dataset.select_dtypes(include=['category']).columns
        if not category_columns.empty:
            dataset[category_columns] = dataset[category_columns].astype('object')

        # Normalizar los nombres de las columnas
        dataset.columns = [col.replace(" ", "_").replace(".", "_") for col in dataset.columns]

        # Preprocesar el dataset
        dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        st.session_state['data'] = dataset

        # Mostrar estadísticas del dataset
        st.success("¡Dataset cargado y preprocesado exitosamente!")
        st.markdown(f"El dataset contiene **{dataset.shape[0]}** filas y **{dataset.shape[1]}** columnas después del preprocesamiento.")
        st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
        if dropped_columns:
            st.markdown(f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**.")
        else:
            st.markdown("No se eliminaron columnas por valores faltantes.")

        # Mostrar una vista previa del dataset
        st.write("Vista previa del dataset:")
        st.dataframe(dataset.head(10))

        # Habilitar el siguiente paso
        st.session_state['step_2_enabled'] = True
