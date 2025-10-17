# 🧠 Mental-Health-Analyzer

**Description:**  
Classifies mental health status from text (sentence) using a BERT-based NLP model. Supports both single text input and batch CSV predictions with probability visualization. The model is trained on mental health-related text datasets and predicts categories like Anxiety, Depression, Stress, etc.

**Dataset Details:**  
- **Source:** Public mental health datasets from social media posts, forums, or research datasets.  
- **Number of samples:** ~50,000+ 
- **Classes:**  
   - Normal
   - Depression
   - Suicidal
   - Anxiety
   - Stress
   - Bi-Polar
   - Personality Disorder
- **Download dataset:** [[Link to dataset]](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  

**Features:**  
1. **Single text input** – Enter a sentence and get the predicted mental health status with probabilities.  
2. **Batch CSV predictions** – Upload a CSV file with multiple sentences to get predictions for each row, including probability visualization.  

**Demo:**  
<img width="1892" height="845" alt="image" src="https://github.com/user-attachments/assets/001c1c2e-ff60-43ff-b2db-1a10634f1c42" />

<img width="1896" height="850" alt="image" src="https://github.com/user-attachments/assets/97366cd8-2b84-4d00-8745-2f62ed924a2e" />

<img width="1877" height="852" alt="image" src="https://github.com/user-attachments/assets/263f19bb-8bb5-4f65-a7b7-a596d2555586" />
- 

<img width="1176" height="305" alt="image" src="https://github.com/user-attachments/assets/662897cd-9e9b-4ebc-8464-d13b2a4e430a" />



**Usage:**  
1. **Download the model Weights(417 mb) from my drive:**
```bash
python download_model.py
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```
streamlit run app.py
```


**Notes:**
- Ensure you have installed the required packages from requirements.txt.
- The model will be saved to the specified path in download_model.py and used by the app for inference.

## Limitations of the Model

### 1. Domain-Specific Data (small scope)
- The model is trained on specific mental health datasets, such as [Kaggle’s Mental Health Sentiment dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health).  
- Performance may degrade on text from other domains, languages, or platforms not seen during training.

### 2. Limited Class Coverage
- Currently predicts only the predefined classes: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, Suicidal.  
- Does not account for other mental health conditions or nuanced emotional states.

### 3. Ambiguous or Sarcastic Text
- The model detects sarcasm at some extent but may fail to correctly classify sarcasm, humor, or ambiguous statements.

### 4. Language Limitation
- Trained primarily on English text; performance on other languages or heavily informal/slang text may be poor.

## Most Imp : 
    it need 2-3 line sentence to understand context better.
### Example : 
- "i am giving up."
     ```may predict normal```
- "I’ve tried everything I could think of, but nothing works. Honestly, I feel like I’m giving up on everything."
    - ```understands NLP and predict Depression/sucidal```
