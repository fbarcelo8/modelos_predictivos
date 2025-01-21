def detect_variable_type(dataset, target):
    """
    Detecta automáticamente el tipo de variable target.
    """
    unique_values = dataset[target].nunique()
    if unique_values < 10:
        if unique_values == 2:
            return "Categórica Binaria"
        else:
            return "Categórica No Binaria"
    else:
        return "Numérica"
