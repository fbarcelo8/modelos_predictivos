
import pandas as pd

def preprocess_dataset(dataset, missing_threshold=0.3):
    """
    Limpia y preprocesa el dataset eliminando duplicados y columnas con un porcentaje alto
    de valores faltantes. Los valores `NaN` permanecen intactos.

    Parameters:
    - dataset: pd.DataFrame, dataset a procesar.
    - missing_threshold: float, porcentaje máximo de valores faltantes permitido (entre 0 y 1).

    Returns:
    - dataset_cleaned: pd.DataFrame, dataset limpio.
    - duplicates_removed: int, número de registros duplicados eliminados.
    - dropped_columns: list, nombres de las columnas eliminadas por exceso de valores faltantes.
    """
    # Eliminar duplicados
    dataset_no_duplicates = dataset.drop_duplicates()
    duplicates_removed = len(dataset) - len(dataset_no_duplicates)

    # Identificar columnas con más del umbral de valores faltantes
    missing_percentage = dataset_no_duplicates.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index.tolist()

    # Eliminar columnas con muchos valores faltantes
    dataset_cleaned = dataset_no_duplicates.drop(columns=columns_to_drop)

    # Asegurar que las columnas categóricas y de tipo object sean consistentes
    for col in dataset_cleaned.select_dtypes(include=['object', 'category']):
        dataset_cleaned[col] = dataset_cleaned[col].astype(str)

    # Dejar los valores NaN intactos para columnas numéricas y categóricas

    return dataset_cleaned, duplicates_removed, columns_to_drop



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

    # Filtrar columnas que son de tipo numérico con decimales
    df = df.select_dtypes(exclude=['float64'])

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
