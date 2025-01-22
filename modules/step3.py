
import streamlit as st
from modules.utils import detect_variable_type

def step_3():
    if not st.session_state.get('step_3_and_4_enabled', False):
        return

    st.header("Paso 3: Detección Automática del Tipo de Variable")
    target = st.session_state['target']
    dataset = st.session_state['data']
    target_type = detect_variable_type(dataset, target)
    st.write(f"La variable seleccionada '{target}' se detectó como: **{target_type}**")

    # Almacenar el tipo de variable en el estado de sesión
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
