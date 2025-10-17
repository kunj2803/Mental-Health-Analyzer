import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------------------
# Config
# ------------------------------
# MODEL_PATH = r"D:\kunj\VS Code\Mental_Health\Final\model"
MODEL_PATH = r"model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load model and tokenizer
# ------------------------------
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ------------------------------
# Label mapping and colors
# ------------------------------
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

label_colors = {
    "Anxiety": "#FFD966",
    "Bipolar": "#FFB266",
    "Depression": "#FF6F91",
    "Normal": "#6FCF97",
    "Personality disorder": "#B399FF",
    "Stress": "#FFA07A",
    "Suicidal": "#FF7F7F"
}
