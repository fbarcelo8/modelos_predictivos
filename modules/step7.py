
import streamlit as st
from sklearn.metrics import mean_squared_error, accuracy_score

def step_7():
    if not st.session_state.get('step_7_enabled', False):
        return

    st.header("Paso 7: Evaluación del Modelo")
    model = st.session_state['trained_model']
    dataset = st.session_state['data']
    fixed_predictors = st.session_state['fixed_predictors']
    candidate_predictors = st.session_state['candidate_predictors']
    target = st.session_state['target']
    target_type = st.session_state['target_type']

    predictors = fixed_predictors + candidate_predictors

    X = dataset[predictors]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if target_type == "Numérica":
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Diseño personalizado para las métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">MSE</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(mse),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">MAE</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(mae),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h3 style="margin: 0; font-size: 1.2rem; font-weight: 600;">R²</h3>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(r2),
                unsafe_allow_html=True,
            )

    else:  # Regresión Logística
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Obtener etiquetas únicas y usar la etiqueta positiva seleccionada en el paso 2
        pos_label = st.session_state.get('pos_label', None)

        # Calcular la matriz de confusión con nombres dinámicos
        unique_labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        # Crear DataFrame con nombres dinámicos para las columnas e índices
        cm_df = pd.DataFrame(
            cm,
            columns=[f"Predicción {label}" for label in unique_labels],
            index=[f"Real {label}" for label in unique_labels]
        )

        # Agregar totales en filas y columnas
        cm_df["Total Real"] = cm_df.sum(axis=1)
        cm_df.loc["Total Predicción"] = cm_df.sum(axis=0)
        
        # Mostrar la matriz de confusión
        st.markdown("**Matriz de Confusión**")
        st.table(cm_df)

        # Calcular métricas de clasificación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        specificity = recall_score(y_test, y_pred, pos_label=unique_labels[0])
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        # Diseño personalizado para métricas de regresión logística
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Exactitud (Accuracy)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(accuracy),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Precisión (Precision)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(precision),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Recall (Sensibilidad)</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(recall),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">Especificidad</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(specificity),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">F1-Score</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(f1),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4 style="margin: 0; font-size: 1.2rem; font-weight: 600;">AUC</h4>
                    <p style="margin: 0; font-size: 1rem; font-weight: 500;">{:.2f}</p>
                </div>
                """.format(auc),
                unsafe_allow_html=True,
            )
    
        # Curva ROC
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=pos_label)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("Curva ROC")
        ax.set_xlabel("1 - Especificidad")
        ax.set_ylabel("Sensibilidad")
        ax.legend()
        st.pyplot(fig)
      
