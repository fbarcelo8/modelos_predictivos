
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
            st.session_state.clear()
        st.session_state['uploaded_file_name'] = file_name
        
        file_extension = file_name.split(".")[-1]
        
        # Manejar errores en la carga del archivo
        try:
            if file_extension == "csv":
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return

        # Verificar si el DataFrame está vacío
        if dataset.empty:
            st.error("El archivo cargado no contiene datos válidos.")
            return
        
        # Procesar el dataset
        dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        
        # Verificar si el DataFrame sigue siendo válido después del preprocesamiento
        if dataset.empty:
            st.error("El dataset quedó vacío después del preprocesamiento. Revisa los valores faltantes o duplicados.")
            return

        # Guardar dataset en el estado de sesión
        st.session_state['data'] = dataset
        
        # Mensajes de éxito
        st.success("¡Dataset cargado y preprocesado exitosamente!")
        st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
        if dropped_columns:
            st.markdown(
                f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**."
            )
        else:
            st.markdown("No se eliminaron columnas por valores faltantes.")
        
        # Validar y mostrar el dataset
        if not dataset.empty:
            st.write("Vista previa del dataset preprocesado:")
            st.table(dataset)
        else:
            st.warning("El dataset no contiene datos para mostrar después del preprocesamiento.")
        
        # Habilitar el siguiente paso
        st.session_state['step_2_enabled'] = True
