import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned BERT model and tokenizer
model_path = './fine-tuned-bert-model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset to label
data_to_label = pd.read_csv('full_data.csv')

# Tokenize the comments and add special tokens
input_ids = []
attention_masks = []
for comment in data_to_label['Comment']:
    encoded_dict = tokenizer.encode_plus(
                        comment,                      # Comment to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens
                        max_length = 512,           # Pad or truncate all comments to 512 tokens.
                        padding = 'max_length',
                        truncation=True,
                        return_attention_mask = True,   # Create attention masks
                        return_tensors ='pt',     # Return PyTorch tensors
                   )
    
    # Add the encoded comment and its attention mask to the lists
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists totensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Create the dataset
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)

# Set the batch size and create the data loader
batch_size = 8
data_loader = torch.utils.data.DataLoader(
            dataset, # The data samples.
            batch_size = batch_size # Inference with this batch size.
        )

# Set the model to evaluation mode
model.eval()

# Loop over batches and generate predictions
all_predictions = []
for batch in data_loader:
    # Move the batch to the device
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_masks = batch
    
    # Disable gradient calculations
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_masks)
        
        # Get predicted labels
        _, predicted = torch.max(outputs[0], 1)
        
        # Convert tensor to list and append to all_predictions
        predicted = predicted.tolist()
        all_predictions.extend(predicted)

# Add the predicted labels to the dataset
data_to_label['label'] = all_predictions

# Save the labeled dataset
data_to_label.to_csv('labeled_data.csv', index=False)