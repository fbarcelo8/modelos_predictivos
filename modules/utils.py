
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
