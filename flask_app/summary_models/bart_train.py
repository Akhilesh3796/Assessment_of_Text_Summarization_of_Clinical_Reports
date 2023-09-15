import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load the data
data_path = "Bio_summarized_data.csv"
data = pd.read_csv(data_path)

# Split data into input text and summaries
input_text = data['Input_Text']
summaries = data['Summaries']

# Split data into training and test sets
train_input_text, test_input_text, train_summaries, test_summaries = train_test_split(
    input_text, summaries, test_size=0.2, random_state=42)

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model.to('cuda')

# Tokenize and prepare DataLoader for training data
def tokenize_data(input_text, summaries, max_length=512):
    inputs = tokenizer(list(input_text), max_length=max_length, truncation=True, padding='max_length', return_tensors='pt', add_prefix_space=True)
    labels = tokenizer(list(summaries), max_length=max_length, truncation=True, padding='max_length', return_tensors='pt', add_prefix_space=True)
    return inputs, labels

train_input_data, train_labels = tokenize_data(train_input_text, train_summaries)
train_dataset = TensorDataset(train_input_data['input_ids'], train_input_data['attention_mask'], train_labels['input_ids'], train_labels['attention_mask'])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Tokenize and prepare DataLoader for test data
test_input_data, test_labels = tokenize_data(test_input_text, test_summaries)
test_dataset = TensorDataset(test_input_data['input_ids'], test_input_data['attention_mask'], test_labels['input_ids'], test_labels['attention_mask'])
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels_input_ids, labels_attention_mask = [item.to('cuda') for item in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_input_ids, decoder_attention_mask=labels_attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Training complete!")

# Save the trained model
model_save_path = "trained_bart_model"
model.save_pretrained(model_save_path)
print(f"Model saved at {model_save_path}")

# Evaluate the model on the test data (optional)
model.eval()
# Add evaluation code as needed for your specific use case
