
import streamlit as st
import pandas as pd
from modules.preprocessing import preprocess_dataset, convert_column_type
from modules.utils import reset_steps, detect_variable_type

def step_1():
    st.header("Paso 1: Subida de Archivo")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        if 'uploaded_file_name' in st.session_state and st.session_state['uploaded_file_name'] != file_name:
            st.session_state.clear()
            st.session_state['uploaded_file_name'] = file_name
        elif 'uploaded_file_name' not in st.session_state:
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

        if dataset.empty:
            st.error("El archivo cargado está vacío. Por favor, sube un archivo válido.")
            return

        # Normalizar los nombres de las columnas
        dataset.columns = [col.replace(" ", "_").replace(".", "_") for col in dataset.columns]
        st.session_state['data'] = dataset  # Guardamos el dataset original en el estado

        # Preprocesar el dataset
        dataset, duplicates_removed, dropped_columns = preprocess_dataset(dataset)
        st.session_state['data'] = dataset

        # Detectar los tipos de variables automáticamente
        if 'variable_types' not in st.session_state:
            st.session_state['variable_types'] = {col: detect_variable_type(dataset, col) for col in dataset.columns}

        st.success("¡Dataset cargado exitosamente!")
        st.dataframe(st.session_state['data'])
        st.markdown(f"El dataset contiene **{dataset.shape[0]}** filas y **{dataset.shape[1]}** columnas.")
        st.markdown(f"Se han eliminado **{duplicates_removed}** registros duplicados.")
        if dropped_columns:
            st.markdown(f"Se han eliminado las siguientes columnas por tener más del 30% de valores faltantes: **{', '.join(dropped_columns)}**.")
        else:
            st.markdown("No se eliminaron columnas por valores faltantes.")

        # Mostrar la tabla interactiva con los tipos de variables
        st.subheader("Tipos de Variables")
        for col in dataset.columns:
            new_type = st.selectbox(
                f"Tipo de '{col}'",
                ["Numérica", "Categórica Binaria", "Categórica No Binaria"],
                index=["Numérica", "Categórica Binaria", "Categórica No Binaria"].index(st.session_state['variable_types'][col]),
                key=f"select_{col}"  # Clave única para evitar conflictos
            )

            # Aplicar conversión inmediata si el usuario cambia el tipo
            if new_type != st.session_state['variable_types'][col]:
                st.session_state['variable_types'][col] = new_type
                convert_column_type(st.session_state['data'], col, new_type)  # Aplicar conversión

        # Botón para confirmar los tipos de las variables
        if st.button("Confirmar tipos de las variables"):
            st.session_state['types_confirmed'] = True  # Marcar como confirmados

        # Mostrar vista previa del dataset actualizado solo si se ha confirmado
        if st.session_state.get('types_confirmed', False):
            st.write("Vista previa del dataset con tipos confirmados:")
            st.dataframe(st.session_state['data'])

            # Habilitar el siguiente paso
            st.session_state['step_2_enabled'] = True
