def step_5():
    if not st.session_state.get('step_5_enabled', False):
        return

    st.header("Paso 5: Selección de Variables Predictoras")
    dataset = st.session_state['data']
    target = st.session_state['target']

    # Inicializar estado si no existe
    if 'fixed_predictors' not in st.session_state:
        st.session_state['fixed_predictors'] = []
    if 'candidate_predictors' not in st.session_state:
        st.session_state['candidate_predictors'] = []

    # Obtener todas las variables predictoras disponibles
    available_predictors = [col for col in dataset.columns if col != target]

    if dataset is not None:
        # Evaluar columnas identificadoras
        resultado_identificadores = evaluar_identificadores(dataset)

        # Filtrar las columnas identificadoras
        identificadoras = resultado_identificadores[resultado_identificadores['Clasificación'] == 'Identificadora']['Columna'].tolist()
        posibles_identificadoras = resultado_identificadores[resultado_identificadores['Clasificación'] == 'Posible identificadora']['Columna'].tolist()

        # Eliminar las columnas identificadoras del dataset antes de continuar
        dataset = dataset.drop(columns=identificadoras, errors='ignore')

        # Guardar el dataset filtrado en la sesión bajo el mismo nombre
        st.session_state['data'] = dataset

        # Actualizar la lista de variables predictoras disponibles después de la eliminación
        available_predictors = [col for col in dataset.columns if col != target]

        # Mostrar un mensaje de éxito con las columnas eliminadas
        if identificadoras:
            st.success(f"Se han eliminado {len(identificadoras)} columnas identificadoras del análisis: {', '.join(identificadoras)}")
        else:
            st.info("No se encontraron columnas identificadoras para eliminar.")

        # Advertir sobre las posibles identificadoras
        if posibles_identificadoras:
            st.warning(f"Las siguientes columnas han sido identificadas como posibles identificadoras y pueden afectar al modelo si se seleccionan: {', '.join(posibles_identificadoras)}.\n\nTen en cuenta que si se usan variables identificadoras como posibles predictoras, el modelo tiene altas probabilidades de dar error, por lo que se recomienda deseleccionarlas.")

    # Variables actualmente seleccionadas
    fixed_predictors_selected = st.session_state['fixed_predictors']
    candidate_predictors_selected = st.session_state['candidate_predictors']

    # Crear listas dinámicas excluyendo las seleccionadas en la otra lista
    fixed_options = [col for col in available_predictors if col not in candidate_predictors_selected]
    candidate_options = [col for col in available_predictors if col not in fixed_predictors_selected]

    # Selección de variables predictoras fijas
    st.markdown("**Selecciona las variables predictoras fijas**")
    fixed_predictors = st.multiselect(
        "Selecciona las Variables Predictoras Fijas",
        fixed_options,
        default=fixed_predictors_selected
    )

    # Selección de variables predictoras candidatas
    st.markdown("**Selecciona las variables predictoras candidatas**")
    candidate_predictors = st.multiselect(
        "Selecciona las Variables Predictoras Candidatas",
        candidate_options,
        default=candidate_predictors_selected
    )

    # Verificar si las selecciones han cambiado
    if set(fixed_predictors) != set(fixed_predictors_selected) or set(candidate_predictors) != set(candidate_predictors_selected):
        st.session_state['fixed_predictors'] = fixed_predictors
        st.session_state['candidate_predictors'] = candidate_predictors
        reset_steps("step_5")
        st.rerun()  # Reiniciar la interfaz para actualizar dinámicamente

    if not fixed_predictors and not candidate_predictors:
        st.warning("Selecciona al menos una variable predictora.")
        return

    # Opciones para manejar valores faltantes
    st.subheader("Manejo de Valores Faltantes")
    handle_missing = st.radio(
        "¿Cómo deseas manejar los valores faltantes en las variables predictoras?",
        options=["Eliminar filas con valores faltantes", "Imputar valores faltantes"],
        index=0
    )

    if st.button("Confirmar Variables Predictoras"):
        selected_predictors = fixed_predictors + candidate_predictors

        if handle_missing == "Eliminar filas con valores faltantes":
            # Eliminar registros con valores faltantes
            missing_rows_before = len(dataset)
            dataset_cleaned = dataset.dropna(subset=selected_predictors)
            missing_rows_after = len(dataset_cleaned)
            rows_removed = missing_rows_before - missing_rows_after

            if rows_removed > 0:
                st.warning(f"Se han eliminado **{rows_removed}** registros en total por valores faltantes.")

                # Calcular cuántos registros fueron eliminados por cada variable
                impact_table = pd.DataFrame({
                    "Variable": selected_predictors,
                    "Registros Eliminados": [dataset[col].isnull().sum() for col in selected_predictors],
                    "Porcentaje Eliminado (%)": [(dataset[col].isnull().sum() / len(dataset) * 100).round(2) for col in selected_predictors]
                }).sort_values(by="Registros Eliminados", ascending=False)

                # Mostrar la tabla ordenada
                st.markdown("**Impacto de las Variables Predictoras en la Eliminación de Registros**")
                st.write("Cantidad de registros eliminados por valores faltantes en cada variable (ordenado de mayor a menor):")
                st.dataframe(impact_table)
            else:
                st.info("No se eliminaron registros por valores faltantes en las variables predictoras seleccionadas.")
        else:
            st.warning("Función todavía no disponible")
            st.stop()

        # Actualizar el estado global
        st.session_state['fixed_predictors'] = fixed_predictors
        st.session_state['candidate_predictors'] = candidate_predictors
        st.session_state['data'] = dataset_cleaned
        st.session_state['step_6_enabled'] = True
        st.session_state['step_7_enabled'] = True
        st.session_state['step_9_enabled'] = True

        st.success("Variables predictoras seleccionadas y datos preprocesados correctamente.")
