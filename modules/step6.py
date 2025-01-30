
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import statsmodels.api as sm 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.pipeline import Pipeline 
from modules.utils import reset_steps, seleccion_forward_bic
from modules.preprocessing import update_predictor_lists, remove_suffix

def step_6():
    if not st.session_state.get('step_6_enabled', False):
        return

    st.header("Paso 6: Entrenamiento del Modelo")

    dataset = st.session_state['data']
    fixed_predictors = st.session_state['fixed_predictors']
    candidate_predictors = st.session_state['candidate_predictors']
    target = st.session_state['target']
    target_type = st.session_state['target_type']
    train_size = st.session_state.get('train_size', 0.8)
    test_size = 1 - train_size

    # Seleccionar las columnas necesarias del dataset
    columns_to_keep = fixed_predictors + candidate_predictors + [target]

    # Filtrar el dataset con las columnas seleccionadas
    filtered_data = dataset[columns_to_keep].copy().dropna()

    # Preprocesamiento de datos
    dataset_clean = pd.get_dummies(filtered_data, drop_first=True).apply(pd.to_numeric, errors='coerce').dropna()
    
    # Obtener la nueva versión del target después de get_dummies()
    new_target_list, _ = update_predictor_lists(dataset_clean, [target], [])

    # Restaurar el nombre original del target solo si NO es Numérica
    original_target = remove_suffix(new_target_list[0]) if target_type != 'Numérica' else new_target_list[0]

    # Renombrar la columna en dataset_clean
    dataset_clean = dataset_clean.rename(columns={new_target_list[0]: original_target})

    # Actualizar las listas de predictores con las nuevas columnas
    new_fixed_predictors, new_candidate_predictors = update_predictor_lists(dataset_clean, fixed_predictors, candidate_predictors)

    # Determinar el tipo de modelo (lineal o logístico)
    modelo_tipo = 'lineal' if target_type == "Numérica" else 'logistica'

    # Selección de variables después del preprocesamiento
    resultado_seleccion = seleccion_forward_bic(
        dataset_clean,
        new_fixed_predictors,
        new_candidate_predictors,
        target,
        tipo_modelo=modelo_tipo
    )

    st.session_state['data'] = dataset_clean
    st.session_state['selected_features'] = resultado_seleccion['mejores_variables']
    st.success(f"Las variables seleccionadas son: {', '.join(resultado_seleccion['mejores_variables'])}")

    predictors = resultado_seleccion['mejores_variables']
    X = dataset_clean[predictors]
    y = dataset_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Preprocesamiento de datos para el modelo
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="passthrough"
    )

    # Entrenar el modelo
    if target_type == "Numérica":
        st.markdown(
            "**Regresión Lineal:** Se ha dividido el conjunto de datos en {}% entrenamiento y {}% prueba."
            .format(round(train_size * 100), round(test_size * 100))
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])
    else:
        st.markdown(
            "**Regresión Logística:** Se ha dividido el conjunto de datos en {}% entrenamiento y {}% prueba."
            .format(round(train_size * 100), round(test_size * 100))
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

    model.fit(X_train, y_train)
    st.session_state['trained_model'] = model
    st.success("Modelo entrenado con éxito.")

    # Obtener nombres de columnas después del preprocesamiento
    feature_names = preprocessor.get_feature_names_out(input_features=predictors)

    # Eliminar prefijos de los nombres de las variables
    clean_feature_names = [name.replace('num__', '').replace('remainder__', '') for name in feature_names]

    # Calcular coeficientes después del preprocesamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    if target_type == "Numérica":
        X_train_sm = sm.add_constant(X_train_processed)
        ols_model = sm.OLS(y_train, X_train_sm).fit()
        coefficients = ols_model.params.values
        p_values = ols_model.pvalues.values
        variables = ['Intercept'] + clean_feature_names
        coef_df = pd.DataFrame({"Variable": variables, "Coeficiente": coefficients, "p-valor": p_values})
        st.write("**Coeficientes del modelo con p-valores:**")
        st.table(coef_df)
        st.markdown("""
        ### 6.1. Interpretación de los coeficientes en una regresión lineal

        En una **regresión lineal**, los coeficientes ($\\beta$) representan la relación entre las variables explicativas ($X$) y la variable dependiente ($Y$).

        #### A) Forma del modelo
        La ecuación general de una regresión lineal es:
        """, unsafe_allow_html=True)

        st.latex(r"""
        Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
        """)

        st.markdown("""
        - $Y$: Variable dependiente.
        - $\\beta_0$: Intercepto, el valor de $Y$ cuando todas las $X_i = 0$.
        - $\\beta_i$: Coeficiente asociado a $X_i$, que mide el efecto de $X_i$ sobre $Y$.
        - $\\epsilon$: Término de error.


        #### B) Interpretación de $\\beta_i$
        Cada coeficiente $\\beta_i$ indica el **cambio esperado en $Y$** por cada unidad adicional en $X_i$, manteniendo constantes las demás variables.

        ### Casos específicos:
        - Si $\\beta_i > 0$: Aumentar $X_i$ incrementa $Y$.
        - Si $\\beta_i < 0$: Aumentar $X_i$ disminuye $Y$.
        - Si $\\beta_i = 0$: $X_i$ no tiene efecto sobre $Y$.


        #### C) Interpretación del intercepto ($\\beta_0$)
        El intercepto $\\beta_0$ representa el valor esperado de $Y$ cuando todas las $X_i = 0$. 

        - En algunos casos, este valor puede no tener un significado práctico si $X_i = 0$ no es realista.


        #### D) Ejemplo práctico
        Supongamos el modelo:
        """, unsafe_allow_html=True)

        st.latex(r"""
        Y = 3 + 2X_1 - 0.5X_2
        """)

        st.markdown("""
        1. Intercepto ($\\beta_0 = 3$):
        - Si $X_1 = 0$ y $X_2 = 0$, $Y = 3$.
        2. Coeficiente de $X_1$ ($\\beta_1 = 2$):
        - Por cada unidad adicional en $X_1$, $Y$ aumenta en 2 unidades, manteniendo constante $X_2$.
        3. Coeficiente de $X_2$ ($\\beta_2 = -0.5$):
        - Por cada unidad adicional en $X_2$, $Y$ disminuye en 0.5 unidades, manteniendo constante $X_1$.


        #### E) Resumen
        En la regresión lineal:

        - Los coeficientes ($\\beta_i$) miden el cambio esperado en $Y$ por cada unidad adicional en $X_i$, controlando por las demás variables.
        - El intercepto ($\\beta_0$) representa el valor de $Y$ cuando todas las $X_i = 0$.
        """, unsafe_allow_html=True)

    else:
        X_train_sm = sm.add_constant(X_train_processed)
        logit_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
        coefficients = logit_model.params.values
        p_values = logit_model.pvalues.values
        variables = ['Intercept'] + clean_feature_names
        coef_df = pd.DataFrame({"Variable": variables, "Coeficiente": coefficients, "p-valor": p_values})
        st.write("**Coeficientes del modelo con p-valores:**")
        st.table(coef_df)
        st.markdown("""
        ### 6.1. Interpretación de los coeficientes en una regresión logística

        En una **regresión logística**, los coeficientes ($\\beta$) no se interpretan directamente como cambios en la variable dependiente (como en una regresión lineal), sino en términos de probabilidades y razones de probabilidades (**odds ratios**).

        #### A) Forma del modelo
        La regresión logística modela la relación entre las variables explicativas ($X$) y la probabilidad de que ocurra un evento ($P(Y=1)$):

        """, unsafe_allow_html=True)

        st.latex(r"""
        \log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n
        """)

        st.markdown("""
        - El lado izquierdo es el logaritmo de las **odds** (razón de probabilidades).
        - $\\beta_0$ es el intercepto, y $\\beta_i$ son los coeficientes asociados a cada $X_i$.


        #### B) Interpretación de $\\beta_i$
        - Si $\\beta_i > 0$: Aumentar $X_i$ incrementa la probabilidad de $Y=1$.
        - Si $\\beta_i < 0$: Aumentar $X_i$ disminuye la probabilidad de $Y=1$.


        #### C) Odds ratio
        El coeficiente transformado exponencialmente ($e^{\\beta_i}$) representa el cambio multiplicativo en las **odds** por cada unidad adicional de $X_i$:

        """, unsafe_allow_html=True)

        st.latex(r"""
        e^{\beta_i} = \text{Odds ratio (OR)}
        """)

        st.markdown("""
        - Si $e^{\\beta_i} = 1$: $X_i$ no tiene efecto.
        - Si $e^{\\beta_i} > 1$: Cada unidad adicional de $X_i$ **aumenta las odds**.
        - Si $e^{\\beta_i} < 1$: Cada unidad adicional de $X_i$ **disminuye las odds**.


        #### D) Ejemplo práctico
        Supongamos que $\\beta_1 = 0.5$:
        - Las **odds** aumentan en un factor de $e^{0.5} \\approx 1.65$ por cada unidad adicional de $X_1$. 
        - Esto significa que el evento $Y=1$ es 1.65 veces más probable.
        """, unsafe_allow_html=True)
