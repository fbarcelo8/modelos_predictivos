def evaluar_identificadores(df, pesos=None, umbral_identificadora=10, umbral_posible_identificadora=6):
    """
    Evalúa las columnas de un DataFrame para determinar si son identificadoras, 
    posibles identificadoras o no identificadoras, basándose en una serie de criterios ponderados.
    """

    # Si pesos no está definido, asignamos valores por defecto
    if pesos is None:
        pesos = {
            'cardinalidad_relativa': 3,
            'ausencia_nulos': 1,
            'control_duplicados': 2,
            'distribucion_uniforme': 1,
            'correlacion_baja': 1,
            'relacion_combinaciones': 3,
            'compatibilidad_tipos': 1,
            'pocas_repeticiones': 2,
            'consistencia_grupos': 3,
            'monotonicidad': 1
        }

    resultados = []

    for columna in df.columns:
        puntaje = 0

        # Evaluar criterios y aplicar pesos
        if df[columna].nunique() >= 0.5 * len(df):
            puntaje += pesos['cardinalidad_relativa']

        if df[columna].isna().sum() == 0:
            puntaje += pesos['ausencia_nulos']

        if df[columna].value_counts().max() < 0.5 * len(df):
            puntaje += pesos['control_duplicados']

        if df[columna].value_counts(normalize=True).max() < 0.5:
            puntaje += pesos['distribucion_uniforme']

        if pd.api.types.is_numeric_dtype(df[columna]):
            # Filtramos solo las columnas numéricas del DataFrame para evitar errores
            df_numerico = df.select_dtypes(include=['number'])
            
            if columna in df_numerico.columns:
                correlaciones = df_numerico.corrwith(df_numerico[columna]).drop(columna, errors='ignore')
                if correlaciones.abs().max() < 0.1:
                    puntaje += pesos['correlacion_baja']

        if 'columna_referencia' in df.columns:
            if df.groupby('columna_referencia')[columna].nunique().max() == 1:
                puntaje += pesos['relacion_combinaciones']

        if df[columna].dtype in ['int64', 'object']:
            puntaje += pesos['compatibilidad_tipos']

        if 'columna_referencia' in df.columns:
            if df.groupby('columna_referencia')[columna].value_counts().max() <= 5:
                puntaje += pesos['pocas_repeticiones']

        if df[columna].is_unique:
            puntaje += pesos['consistencia_grupos']

        if not (df[columna].is_monotonic_increasing or df[columna].is_monotonic_decreasing):
            puntaje += pesos['monotonicidad']

        if puntaje >= umbral_identificadora:
            clasificacion = 'Identificadora'
        elif puntaje >= umbral_posible_identificadora:
            clasificacion = 'Posible identificadora'
        else:
            clasificacion = 'No identificadora'

        resultados.append({
            'Columna': columna,
            'Puntaje': puntaje,
            'Clasificación': clasificacion
        })

    resultados_df = pd.DataFrame(resultados)

    return resultados_df
