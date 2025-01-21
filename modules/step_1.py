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
