import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from io import BytesIO

# ------------------------------
# Download model weights first
# ------------------------------
st.write("Checking/downloading model weights... ⏳")
import Download_model  # this will execute download_model.py

# ------------------------------
# Load model
# ------------------------------
from utils import load_model, label_map, label_colors, DEVICE,clean_and_lemmatize_text

with st.spinner("Loading the Mental Health model... Please wait..."):
    tokenizer, model = load_model()

# ------------------------------
# Page layout & Dark Theme
# ------------------------------
st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠", layout="wide")

st.markdown("""<style>
#MainMenu, footer, header {visibility: hidden;}
body, .stApp { background-color: #0E1117; color: #FAFAFA; font-size:18px; }
.stTextArea textarea { background-color: #1C1F26 !important; color: #FAFAFA !important; font-size:28px; line-height:1.8; padding:20px; border-radius:10px; }
textarea::placeholder { color: #AAAAAA !important; font-size:28px; font-style:italic; }
.stButton button, .stDownloadButton button { background-color: #2C2F38; color: #FAFAFA; border-radius: 8px; padding: 0.6em 1.2em; font-size:18px; }
.stButton button:hover, .stDownloadButton button:hover { background-color: #444752; }
.stDataFrame { background-color: #1C1F26 !important; color: #FAFAFA !important; font-size:22px; }
.stDataFrame th, .stDataFrame td { color: #FAFAFA !important; font-size:22px; padding:12px 10px; min-width:200px; }
.prob-label { font-weight:bold; font-size:22px; width:auto; min-width:180px; }
.prob-percent { margin-left:10px; font-size:20px; }
</style>""", unsafe_allow_html=True)

st.title("🧠 Mental Health Analyzer")

# ------------------------------
# Single Prediction
# ------------------------------
st.subheader("Single Prediction")
example_text = "I feel restless and anxious all the time."
user_input = st.text_area(
    "Enter your statement here:", 
    height=250,                        
    value=example_text, 
    placeholder="Type something like 'I feel anxious or depressed...'"
)

if st.button("Predict Single"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned_text = clean_and_lemmatize_text(user_input)
        inputs = tokenizer(cleaned_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(probs.argmax())
            pred_label = label_map[pred_id]
            confidence = probs[pred_id]*100

        color = label_colors.get(pred_label, "#FAFAFA")
        st.markdown(f"""
            <div style="font-size:36px; font-weight:bold; margin-top:15px; line-height:1.5;">
                Predicted Status&nbsp;:&nbsp;
                <span style='color:{color};'>{pred_label}</span>
                &nbsp;&nbsp;({confidence:.2f}%)
            </div>
            """, unsafe_allow_html=True)

        for i, label in enumerate(label_map.values()):
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:8px 0;">
                <div class="prob-label">{label}</div>
                <div style="background-color:{label_colors[label]}; width:{probs[i]*100:.2f}%; height:24px; border-radius:6px;"></div>
                <div class="prob-percent">{probs[i]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------
# Batch Prediction from CSV 
# ------------------------------
st.markdown("""<hr style='border:1px solid #444752; margin:30px 0;'>""", unsafe_allow_html=True)

st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with a column 'text'", type=["csv"], key="batch")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        all_preds = []
        for txt in df['text']:
            inputs = tokenizer(str(txt), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred_id = int(logits.argmax(dim=1).cpu().item())
                pred_label = label_map[pred_id]
                all_preds.append(pred_label)

        df['predicted_label'] = all_preds
        st.dataframe(df[['text','predicted_label']], use_container_width=True)

        csv_buffer = BytesIO()
        df[['text','predicted_label']].to_csv(csv_buffer, index=False)
        st.download_button("Download Predictions CSV", data=csv_buffer, file_name="predictions.csv", mime="text/csv")
