import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import ParameterGrid

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
        comment,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
   
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(train_data['label'])

# Create the train dataset
train_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

# Set the batch size and create the data loader
batch_size = 8
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

# Define the hyperparameters and their search space
hyperparameters = {
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'epochs': [3, 4, 5]
}

# Iterate over the hyperparameter combinations
best_score = 0.0
best_hyperparameters = None

for params in ParameterGrid(hyperparameters):
    learning_rate = params['learning_rate']
    epochs = params['epochs']

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.to(device)

    # Set the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            optimizer.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Average training loss: {}".format(avg_train_loss))

    # Evaluate the model using your preferred evaluation metric
    # Replace this code with your evaluation code
    evaluation_score = avg_train_loss

    # Check if this hyperparameter combination is the best so far
    if evaluation_score > best_score:
        best_score = evaluation_score
        best_hyperparameters = params

# Print the best hyperparameters
print("Best hyperparameters: {}".format(best_hyperparameters))