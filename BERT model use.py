# Import necessary libraries
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Load the train dataset
train_data = pd.read_csv('train_data.csv')

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the comments and add special tokens
input_ids = []
attention_masks = []
for comment in train_data['Comment']:
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

# Convert the lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(train_data['label'])

# Create the train dataset
train_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

# Set the batch size and create the data loader
batch_size = 8
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Randomly sample the data during training.
            batch_size = batch_size # Trains with this batch size.
        )

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 4, # Number of output labels.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states= False, # Whether the model returns all hidden-states.
)

# Move the model to the device
model.to(device)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # Learning rate
                  eps = 1e-8 # Adam epsilon
                )

# Train the model
epochs = 4
for epoch in range(epochs):
    # Set the model to training mode
    model.train()
    
    # Set the total loss for this epoch
    total_loss = 0
    
    # Iterate over batches
    for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch)):
        # Move the batch to the device
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, labels = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, 
                        attention_mask=attention_masks, 
                        labels=labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip the norm of the gradients to 1.0 to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
    # Print the average loss for this epoch
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average training loss: {}".format(avg_train_loss))

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-bert-model')

# Save the tokenizer
tokenizer.save_pretrained('./fine-tuned-bert-model')