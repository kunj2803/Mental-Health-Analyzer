import torch
from transformers import BertTokenizer, BertForSequenceClassification
import html
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import nltk
import os

project_dir = os.path.dirname(__file__)
nltk_data_path = os.path.join(project_dir, "nltk_data")

# Add this path so NLTK can find your data
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# POS mapper for lemmatizer
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_and_lemmatize_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""  # just return empty string for now

    # Decode HTML entities
    text = html.unescape(text)

    # Fix encoding issues
    try:
        text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    except:
        pass

    # Remove RTs, mentions, URLs
    text = re.sub(r"\bRT\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep hashtags as words
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove non-alphanumeric junk characters except punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return ""  # keep as empty, will drop later

    # Tokenize and POS tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]

    return " ".join(lemmatized_tokens)

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
