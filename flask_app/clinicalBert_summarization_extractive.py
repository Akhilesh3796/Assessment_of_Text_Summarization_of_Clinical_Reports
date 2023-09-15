import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def clinical_extractive_summary(text):
    
    # Load ClinicalBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    max_summary_length = 100
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
    extractive_summary = '. '.join(filtered_summary_sentences).strip()
    
    return extractive_summary

# Example usage
medical_text = """
Cancer is a complex and multifaceted disease that arises from the uncontrolled growth and division of abnormal cells in the body.
It can affect virtually any tissue or organ and has numerous subtypes, each with its own unique characteristics and challenges.
The development of cancer involves a series of genetic mutations that disrupt the normal regulation of cell growth, division, and death.
Early detection and diagnosis of cancer are crucial for successful treatment and improved patient outcomes.
Advances in medical imaging, molecular biology, and genetic testing have led to the identification of biomarkers and genetic signatures.
Additionally, targeted therapies and immunotherapies have emerged as promising approaches for treating cancer.
Cancer research involves a collaborative effort from scientists, clinicians, and researchers across various disciplines.
The study of cancer biology, genetics, and genomics provides insights into the underlying mechanisms of the disease and potential therapeutic targets.
Clinical trials are conducted to evaluate the safety and efficacy of new treatments and interventions.
Overall, advancements in cancer research and treatment have contributed to improved survival rates and quality of life for individuals affected by cancer.
"""

summary = clinical_extractive_summary(medical_text)
print("Extractive Summary (up to 100 words):")
print(summary)
