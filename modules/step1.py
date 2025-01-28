
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
        
        file_extension = file_name.split(".")[-1]

        # Intentar leer el archivo
        try:
            if file_extension == "csv":
                # Intentar con diferentes codificaciones
                try:
                    dataset = pd.read_csv(uploaded_file)  # Intento inicial con codificación predeterminada (utf-8)
                except UnicodeDecodeError:
                    st.warning("Error de codificación. Probando con 'latin1'.")
                    dataset = pd.read_csv(uploaded_file, encoding="latin1")
            else:
                dataset = pd.read_excel(uploaded_file, engine="openpyxl")  # Usar openpyxl para Excel
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return
        
        # Procesar el dataset
        try:
            dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        except Exception as e:
            st.error(f"Error al procesar el dataset: {e}")
            return

        # Verificar si el dataset está vacío
        if dataset.empty:
            st.error("El dataset quedó vacío después del procesamiento. Verifica el archivo.")
            return

        # Guardar dataset en la sesión
        st.session_state['data'] = dataset
        
        # Mensajes de éxito y detalles del preprocesamiento
        st.success("¡Dataset cargado y preprocesado exitosamente!")
        st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
        if dropped_columns:
            st.markdown(
                f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**."
            )
        else:
            st.markdown("No se eliminaron columnas por valores faltantes.")

        # Mostrar el dataset en la interfaz
        st.write("Vista previa del dataset:")
        st.dataframe(dataset)  # Mostrar la tabla interactiva

        # Habilitar el siguiente paso
        st.session_state['step_2_enabled'] = True
