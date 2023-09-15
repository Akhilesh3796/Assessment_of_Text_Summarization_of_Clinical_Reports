from transformers import AutoTokenizer, AutoModel,BartTokenizer, BartForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import BartTokenizer, BartForSequenceClassification

def calculate_similarity_biobert(text1, text2):

    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input texts
    inputs = tokenizer(text1, text2, padding=True, truncation=True, return_tensors="pt")

    # Pass the input through the BioBERT model
    outputs = model(**inputs)

    # Extract the embeddings from the model outputs
    embeddings = outputs.last_hidden_state.squeeze(0)  # Squeeze the batch dimension

    # Calculate the cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)

    return similarity.item()

def calculate_similarity_clinicalBert(text1, text2):
    # Tokenize the input texts
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer.encode_plus(text1, text2, add_special_tokens=True, truncation=True, max_length=512, padding="longest", return_tensors="pt")

    # Pass the input through the Clinical BERT model
    outputs = model(**inputs)

    # Extract the embeddings from the model outputs
    embeddings = outputs.last_hidden_state.squeeze(0)  # Squeeze the batch dimension

    # Calculate the cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)

    return similarity.item()



def perform_zero_shot_classification(text1, text2):
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

    # Tokenize inputs
    premise = text1
    hypothesis = text2

    # Encode inputs and generate attention masks
    encoding = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Perform classification
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Interpret logits to determine the classification
    probabilities = logits.softmax(dim=1)[0]
    entailment_prob = probabilities[0].item()
    contradiction_prob = probabilities[2].item()

    # Determine predicted classification based on probabilities
    if entailment_prob > contradiction_prob:
        predicted_classification = "Entailment"
    else:
        predicted_classification = "Contradiction"

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Predicted Classification: {predicted_classification}")
    print("")

    return predicted_classification



from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline

def roberta_relationship(premise, hypothesis):
    # Load pre-trained RoBERTa model and tokenizer
    model_name = "textattack/roberta-base-mnli"  # Example model variant
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    # Create a textual entailment pipeline
    entailment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Example label map
    label_map = {
        "LABEL_0": "Entailment",
        "LABEL_1": "Contradiction",
        "LABEL_2": "Neutral"
    }

    result = entailment_pipeline(premise, hypothesis)
    label_id = result[0]['label']
    label_text = label_map[label_id]
    
    return label_text







