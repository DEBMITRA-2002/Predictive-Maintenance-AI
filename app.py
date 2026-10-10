
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 2. LOAD MODEL
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("⚙️ Predictive Maintenance AI")
st.markdown("### Intelligent Machine Failure Prediction")
st.write(
    "Analyze machine sensor readings, estimate failure probability, "
    "and explore the features used by the trained model."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.divider()


# =====================================================
# 3. SIDEBAR
# =====================================================

with st.sidebar:
    st.header("⚙️ Project Information")
    st.write("**Project:** Predictive Maintenance AI")
    st.write("**Dataset:** AI4I 2020 Predictive Maintenance")
    st.write("**Task:** Binary classification")
    st.success("Trained model loaded")

    st.divider()
    st.caption(
        "This is an academic prototype. Model predictions do not "
        "replace machine inspections or industrial safety systems."
    )


# =====================================================
# 4. DASHBOARD TABS
# =====================================================

prediction_tab, explanation_tab, about_tab = st.tabs(
    ["🔍 Prediction", "📊 Model Insights", "ℹ️ About"]
)


# =====================================================
# 5. PREDICTION TAB
# =====================================================

with prediction_tab:
    st.header("Machine Sensor Inputs")
    st.write("Enter the machine's current sensor readings.")

    left, right = st.columns(2)

    with left:
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            help="L, M and H represent machine product-quality variants."
        )

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1
        )

    with right:
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=0,
            max_value=5000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=0.5
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=500,
            value=100,
            step=1
        )

    predict_button = st.button(
        "🔍 Predict Machine Failure",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        try:
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            failure_probability = float(probabilities[1])

            st.divider()
            st.subheader("Prediction Results")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Estimated Failure Probability",
                    f"{failure_probability:.1%}"
                )

            with metric2:
                st.metric(
                    "Predicted Class",
                    "Failure" if prediction == 1 else "No Failure"
                )

            st.write("**Failure probability indicator**")
            st.progress(
                max(0.0, min(1.0, failure_probability))
            )

            if prediction == 1:
                st.error(
                    "⚠️ The model predicts a possible machine failure. "
                    "Inspect the machine and verify its condition."
                )
            else:
                st.success(
                    "✅ The model predicts no failure for these inputs. "
                    "Continue routine monitoring."
                )

            with st.expander("View submitted sensor readings"):
                st.dataframe(input_data, hide_index=True,
                             use_container_width=True)

            with st.expander("View probability details"):
                st.write(
                    f"No-failure probability: {probabilities[0]:.4f}"
                )
                st.write(
                    f"Failure probability: {probabilities[1]:.4f}"
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =====================================================
# 6. MODEL INSIGHTS TAB
# =====================================================

with explanation_tab:
    st.header("📊 Model Feature Importance")
    st.write(
        "Feature importance indicates how much each input feature "
        "contributes to the fitted model's decisions. It does not "
        "establish causation."
    )

    classifier = model.named_steps.get("classifier")
    fitted_preprocessor = model.named_steps.get("preprocessor")

    if (
        classifier is not None
        and fitted_preprocessor is not None
        and hasattr(classifier, "feature_importances_")
    ):
        try:
            feature_names = (
                fitted_preprocessor.get_feature_names_out()
            )
            importance_values = classifier.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_values
            }).sort_values(
                "Importance", ascending=False
            ).reset_index(drop=True)

            st.subheader("Top 10 Important Features")
            top_features = importance_df.head(10).sort_values(
                "Importance", ascending=True
            )

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(
                top_features["Feature"],
                top_features["Importance"]
            )
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance — Selected Model")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Importance Table")
            st.dataframe(
                importance_df.round(4),
                hide_index=True,
                use_container_width=True
            )

        except Exception as exc:
            st.warning(
                "Could not calculate feature importance "
                f"for this model: {exc}"
            )

    else:
        st.info(
            "This selected model does not expose built-in tree "
            "feature importance. Permutation importance can be "
            "calculated separately if needed."
        )


# =====================================================
# 7. ABOUT TAB
# =====================================================

with about_tab:
    st.header("About This Project")

    st.markdown("""
    **Objective**

    Estimate machine failure from sensor measurements using a
    trained machine learning classification pipeline.

    **Input features**
    - Machine type
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear

    **Output**
    - Predicted class: failure or no failure
    - Model-estimated probability of failure

    **Dataset limitation**

    The AI4I 2020 dataset is synthetic. Performance on this dataset
    does not guarantee performance on real industrial equipment.
    """)


# =====================================================
# 8. FOOTER
# =====================================================

st.divider()
st.caption(
    "Predictive Maintenance AI | Machine Learning Academic Project"
)
