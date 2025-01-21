def preprocess_dataset(dataset, missing_threshold=0.3):
    """
    Limpia y preprocesa el dataset eliminando valores nulos y duplicados.
    Además, elimina columnas con un porcentaje alto de valores faltantes.
    
    Parameters:
    - dataset: pd.DataFrame, dataset a procesar.
    - missing_threshold: float, porcentaje máximo de valores faltantes permitido (entre 0 y 1).
    
    Returns:
    - dataset: pd.DataFrame, dataset limpio.
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
    
    return dataset_cleaned, duplicates_removed, columns_to_drop
