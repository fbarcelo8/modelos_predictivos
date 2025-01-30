
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from modules.utils import detect_variable_type, reset_steps


def step_3():
    if not st.session_state.get('step_3_and_4_enabled', False):
        return

    st.header("Paso 3: Confirmación de Tipos de Variables y Distribución de la Variable Target")

    dataset = st.session_state['data'].copy()  # Crear una copia para modificar tipos sin afectar el original
    target = st.session_state['target']
    variable_types = st.session_state['variable_types']  # Tipos seleccionados en step_1

    # Asegurar que cada columna tenga el tipo seleccionado o detectado en step_1()
    for col, new_type in variable_types.items():
        convert_column_type(dataset, col, new_type)  # Aplicar conversión

    # Guardar dataset actualizado con los tipos correctos en session_state
    st.session_state['data'] = dataset  

    # Crear DataFrame con los tipos de variables
    type_info = pd.DataFrame({
        "Variable": dataset.columns,
        "Tipo seleccionado": [variable_types[col] for col in dataset.columns],
        "type": [dataset[col].dtype for col in dataset.columns]
    })

    # Mostrar la tabla con los tipos de variables
    st.subheader("Tipos de Variables Seleccionados")
    st.table(type_info)

    # Almacenar el tipo de variable en el estado de sesión
    target_type = variable_types[target]
    st.session_state['target_type'] = target_type

    if target_type == "Categórica Binaria":
        st.subheader("Distribución de la Variable Target")

        # Contar valores únicos de la variable target
        target_counts = dataset[target].value_counts()
        total = len(dataset)

        # Crear un DataFrame con los valores y porcentajes
        target_distribution = pd.DataFrame({
            "Categoría": target_counts.index.astype(str),
            "Frecuencia": target_counts.values,
            "Porcentaje": (target_counts.values / total) * 100
        })

        # Mostrar la tabla con porcentajes
        st.table(target_distribution.style.format({"Porcentaje": "{:.2f}%"}))

        # Crear un gráfico de barras con los colores personalizados
        colors = ['#A759FE', '#FFFF4B']
        fig, ax = plt.subplots()
        sns.barplot(
            x=target_distribution["Categoría"],
            y=target_distribution["Frecuencia"],
            palette=colors,
            ax=ax
        )
        ax.set_title("Distribución de la Variable Target")
        ax.set_xlabel("Categoría")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    st.session_state['step_4_enabled'] = True
