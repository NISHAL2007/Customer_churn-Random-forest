import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("Random_forest.pkl")

def predict_func(MonthlyCharges, tenure, Contract):
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}  # use this map
    df = pd.DataFrame([{
        "MonthlyCharges": float(MonthlyCharges),
        "tenure": int(tenure),
        "Contract": contract_map[Contract]  # convert string to int code
    }])
    pred = model.predict(df)[0]
    return "Churn" if pred == 1 else "No Churn"


# UI Elements
inputs = [
    gr.Textbox(label="Monthly Charges"),
    gr.Slider(0, 100, value=12, step=1, label="Tenure (months)"),
    gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
]

demo = gr.Interface(
    fn=predict_func,
    inputs=inputs,
    outputs="text",
    title="Telco Customer Churn Prediction",
    description="Predict churn using MonthlyCharges, Tenure (slider), and Contract type.<br><br>",
    # Add custom CSS for blurred background text
    css="""
    body {
        background: #f6f7fb !important;
    }
    .gradio-container:before {
        content: "ROHIT 45";
        position: fixed;
        top: 35%;
        left: 25%;
        font-size: 8vw;
        color: #0004;
        z-index: 0;
        filter: blur(6px);
        font-weight: bold;
        pointer-events: none;
        user-select: none;
        opacity: 0.09;
    }
    """
)

if __name__ == "__main__":
    demo.launch()
