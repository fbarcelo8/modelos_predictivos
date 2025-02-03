
import streamlit as st
from modules.step1 import step_1
from modules.step2 import step_2
from modules.step3 import step_3
from modules.step4 import step_4
from modules.step5 import step_5
from modules.step6 import step_6
from modules.step7 import step_7
from modules.step9 import step_9

# Configuración de la página
st.set_page_config(
    page_title="AutoModeler",  # Título de la pestaña
    page_icon="images/B_butler.png",  # Ruta de la imagen en tu proyecto
    layout="centered",  # Opcional: define el diseño de la página
)

def main():
    # Crear columnas para centrar la imagen
    col1, col2, col3 = st.columns([1, 4, 1])  # Ajusta las proporciones para centrar la imagen
    with col2:
        st.image("images/logo_butler1.png", width=450)

    # Mostrar el título centrado y más grande
    st.markdown(
        """
        <h1 style="text-align: center; font-size: 60px; font-weight: bold; color: black;">
            AutoModeler
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="text-align: center; font-size: 30px; font-weight: bold;">
            Herramienta de Modelos Predictivos Semiautomática
        </h2>
        """,
        unsafe_allow_html=True
    )


    st.error("⚠️ **AVISO**: AutoModeler es actualmente funcional para predecir variables numéricas mediante una **Regresión Lineal** o variables categóricas binarias mediante una **Regresión Logística**. Próximamente habrá más funcionalidades disponibles.")
    
    st.write("Sube tu dataset y sigue los pasos para analizarlo y entrenar un modelo.")

    step_1()
    step_2()
    step_3()
    step_4()
    step_5()
    step_6()
    step_7()
    step_9()

if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    main()
    
