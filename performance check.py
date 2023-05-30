import pandas as pd
import torch
from torch.utils.data.sampler import SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

# Load the test dataset
test_data = pd.read_csv('train_data.csv')

# Define the device to use for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine-tuned-bert-model')
tokenizer = BertTokenizer.from_pretrained('./fine-tuned-bert-model', do_lower_case=True)

# Tokenize the comments and add special tokens
input_ids = []
attention_masks = []
for comment in test_data['Comment']:
    encoded_dict = tokenizer.encode_plus(
                        comment,                      # Comment to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens
                        max_length = 512,           # Pad or truncate all comments to 512 tokens.
                        padding = 'max_length',
                        truncation=True,
                        return_attention_mask = True)   

    # Add the encoded comment and its attention mask to the lists
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists to tensors
input_ids = torch.stack([torch.tensor(x) for x in input_ids])
attention_masks = torch.stack([torch.tensor(x) for x in attention_masks])
labels = torch.tensor(test_data['label'])

# Create the test dataset
test_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

# Set the batch size and create the data loader
batch_size = 8
test_dataloader = DataLoader(
            test_dataset,  # The test samples.
            sampler = SequentialSampler(test_dataset), # Sequentially sample the data during testing.
            batch_size = batch_size # Test with this batch size.
        )

# Set the model to evaluation mode
model.eval()

# Initialize some variables
predictions, true_labels = [], []

# Iterate over batches
for batch in test_dataloader:
    # Move the batch to the device
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_masks, labels = batch
    
    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, 
                        attention_mask=attention_masks)
        
    # Get the predicted label and true label
    logits = outputs[0]
    predictions.extend(torch.argmax(logits, axis=1).tolist())
    true_labels.extend(labels.tolist())

# Replace NaN values in true_labels with 0
true_labels = np.nan_to_num(true_labels, nan=0)

# Calculate the precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=1)

# Print the precision, recall, and F1 score
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Plot the learning rate chart
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # Learning rate
                  eps = 1e-8 # Adam epsilon
                )

epochs = 4
total_steps = len(test_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

# Plot the learning rate versus the training steps
plt.plot(range(total_steps), [scheduler.step() for _ in range(total_steps)])

plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()