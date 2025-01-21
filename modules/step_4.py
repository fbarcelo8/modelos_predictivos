def step_4():
    if not st.session_state.get('step_4_enabled', False):
        return

    st.header("Paso 4: Selección del Modelo")
    target_type = st.session_state['target_type']
    previous_model = st.session_state.get('model', None)
    previous_train_size = st.session_state.get('train_size', 0.8)

    if target_type == "Numérica":
        # Lista de modelos disponibles para variables numéricas
        models = [
            "Regresión Lineal",
            "Regresión No Lineal",
            "Árbol de Decisión",
            "Random Forest",
            "BaggingRegressor",
            "XGBoost",
            "Red Neuronal"
        ]
        selected_model = st.selectbox(
            "Selecciona el modelo que deseas utilizar:",
            models,
            index=models.index(previous_model) if previous_model in models else 0  # Regresión Lineal predeterminado
        )
    else:
        # Para variables categóricas, mantenemos Regresión Logística
        selected_model = "Regresión Logística"
        st.info(f"Modelo seleccionado: **{selected_model}**")

    if selected_model != previous_model:
        reset_steps("step_3_and_4")

    # Selección del tamaño del conjunto de entrenamiento
    st.subheader("Seleccionar el tamaño de conjunto de entrenamiento")
    train_size = st.slider("Selecciona el tamaño del conjunto de entrenamiento", 
                           min_value=0.05, max_value=0.95, value=previous_train_size, step=0.05)
    
    # Si el modelo o train_size cambian, se resetean los pasos posteriores
    if selected_model != previous_model or train_size != previous_train_size:
        reset_steps("step_3_and_4")

    if st.button("Confirmar Modelo"):
        st.session_state['model'] = selected_model
        st.session_state['train_size'] = train_size

        if selected_model != "Regresión Lineal" and target_type == "Numérica":
            st.warning(f"El modelo **{selected_model}** está en desarrollo y no está disponible actualmente.")
            st.session_state['step_5_enabled'] = False  # Asegúrate de deshabilitar el siguiente paso
        else:
            st.success("Modelo seleccionado.")
            st.session_state['step_5_enabled'] = True
