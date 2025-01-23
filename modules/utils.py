
import streamlit as st
import pandas as pd


def reset_steps(from_step):
    """
    Resetea todos los pasos posteriores al paso dado.
    """
    steps_to_reset = {
        "step_2": ["step_3_and_4_enabled", "step_5_enabled", "step_6_enabled", "step_7_enabled", "step_9_enabled"],
        "step_3_and_4": ["step_5_enabled", "step_6_enabled", "step_7_enabled", "step_9_enabled"],
        "step_5": ["step_6_enabled", "step_7_enabled", "step_9_enabled"],
    }
    
    for step in steps_to_reset.get(from_step, []):
        if step in st.session_state:
            del st.session_state[step]

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

def seleccion_forward_bic(df, variables_fijas, variables_candidatas, objetivo, tipo_modelo='lineal'):
    """
    Selecciona las mejores variables predictoras para un modelo de regresión usando el método forward selection
    basado en el BIC.

    Parámetros:
        df (pd.DataFrame): DataFrame preprocesado (sin valores categóricos, solo numéricos).
        variables_fijas (list): Variables que siempre deben estar en el modelo.
        variables_candidatas (list): Variables candidatas a evaluar.
        objetivo (str): Nombre de la variable objetivo.
        tipo_modelo (str): 'lineal' para regresión lineal, 'logistica' para regresión logística.

    Retorna:
        dict: Conjunto óptimo de variables y su BIC.
    """

    # Inicializar el modelo con las variables fijas
    variables_modelo = list(variables_fijas)
    mejores_variables = list(variables_fijas)
    mejor_bic = float('inf')

    # Lista de variables candidatas disponibles
    candidatas_restantes = list(variables_candidatas)

    while candidatas_restantes:
        bic_min_actual = float('inf')
        mejor_variable = None

        for variable in candidatas_restantes:
            # Probar el modelo con la variable candidata actual
            variables_prueba = variables_modelo + [variable]
            X = df[variables_prueba]
            X = sm.add_constant(X)  # Agregar la constante

            y = df[objetivo]

            try:
                # Ajustar el modelo según el tipo de regresión
                if tipo_modelo == 'lineal':
                    modelo = sm.OLS(y, X).fit()
                elif tipo_modelo == 'logistica':
                    modelo = sm.Logit(y, X).fit(disp=0)
                else:
                    raise ValueError("El tipo_modelo debe ser 'lineal' o 'logistica'.")

                bic_actual = modelo.bic

                # Evaluar si el BIC es el mejor hasta ahora
                if bic_actual < bic_min_actual:
                    bic_min_actual = bic_actual
                    mejor_variable = variable
            except Exception as e:
                print(f"Error al procesar la variable {variable}: {e}")
                continue

        # Si el nuevo BIC es mejor, agregar la variable al modelo
        if mejor_variable and bic_min_actual < mejor_bic:
            mejor_bic = bic_min_actual
            variables_modelo.append(mejor_variable)
            candidatas_restantes.remove(mejor_variable)
        else:
            break  # Si no mejora el BIC, se detiene

    return {
        'mejores_variables': variables_modelo,
        'mejor_BIC': mejor_bic
    }
