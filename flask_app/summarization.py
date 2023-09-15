from summarizer import TransformerSummarizer
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from rouge_score import rouge_scorer
# GPT-2 model for summarization





# BART model and tokenizer for summarization

# T5 model and tokenizer for summarization
def generate_summary_gpt2(text_primary,text_secondary):
    text =text_primary + " " + text_secondary
    GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    summary_primary = GPT2_model(text_primary, min_length=20,max_length=200)
    summary_secondary = GPT2_model(text_secondary, min_length=20,max_length=200)
    summary = summary_primary + " " + summary_secondary
    rouge_gpt2 = calculate_rouge(text, summary)
    return summary, rouge_gpt2

def generate_summary_bart(text_primary,text_secondary):
    text =text_primary + " " + text_secondary
    
    BART_model_name = "facebook/bart-large-cnn"
    BART_tokenizer = BartTokenizer.from_pretrained(BART_model_name)
    BART_model = BartForConditionalGeneration.from_pretrained(BART_model_name)

    

    inputs_primary = BART_tokenizer(text_primary, return_tensors="pt", max_length=1024, truncation=True)
    
    inputs_secondary = BART_tokenizer(text_secondary, return_tensors="pt", max_length=1024, truncation=True)
    summary_primary_ids = BART_model.generate(inputs_primary.input_ids, max_length=200, min_length=10, num_beams=4, early_stopping=True)
    summary_secondary_ids = BART_model.generate(inputs_secondary.input_ids, max_length=200, min_length=10, num_beams=4, early_stopping=True)
    
    summary_primary = BART_tokenizer.decode(summary_primary_ids[0], skip_special_tokens=True)
    summary_secondary = BART_tokenizer.decode(summary_secondary_ids[0], skip_special_tokens=True)
    
    summary=summary_primary + " " + summary_secondary
    rouge_bart = calculate_rouge(text, summary)
    return summary , rouge_bart

def generate_summary_T5_1(text_primary,text_secondary):
    text =text_primary + " " + text_secondary
    
    model_name = "t5-base"  # You can also use "t5-base", "t5-large", etc.
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_primary_ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_primary_ids = t5_model.generate(input_primary_ids, max_length=200, num_beams=4, early_stopping=True)
    summary_primary = t5_tokenizer.decode(summary_primary_ids[0], skip_special_tokens=True)
    
    input_secondary_ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_secondary_ids = t5_model.generate(input_secondary_ids, max_length=200, num_beams=4, early_stopping=True)
    summary_secondary = t5_tokenizer.decode(summary_secondary_ids[0], skip_special_tokens=True)
    
    summary=summary_primary + " " + summary_secondary

    rouge_t5 = calculate_rouge(text, summary)
    return summary , rouge_t5


def clinical_summary(text_primary,text_secondary):
    text =text_primary + " " + text_secondary
    
    # Load ClinicalBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    max_summary_length = 200
    num_summary_sentences = 2
    # Tokenize and generate embeddings
    sentences = text.split(".")
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Average pooling
            sentence_embeddings.append(embeddings[0])  # Remove the extra dimension

    # Calculate cosine similarity
    cosine_similarities = np.dot(np.array(sentence_embeddings), np.array(sentence_embeddings).T)
    np.fill_diagonal(cosine_similarities, 0)  # Set diagonal to 0 to avoid selecting the same sentence

    # Select top sentences based on cosine similarity
    summary_indices = np.argsort(cosine_similarities.mean(axis=1))[::-1][:num_summary_sentences]

    # Filter sentences to ensure max length of 100 words
    filtered_summary_sentences = []
    summary_length = 0

    for idx in summary_indices:
        sentence_length = len(sentences[idx].split())
        if summary_length + sentence_length <= max_summary_length:
            filtered_summary_sentences.append(sentences[idx])
            summary_length += sentence_length
        else:
            break

    # Generate the extractive summary
    summary = '. '.join(filtered_summary_sentences).strip()

    rouge_clinical = calculate_rouge(text, summary)

    return summary , rouge_clinical

def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

