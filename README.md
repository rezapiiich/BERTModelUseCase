# Classification using BERT model full documentation
# Reza Farshchi 
# May 2023

## Table of contents:
* Abstract
* Model Selection
* Data Preprocessing
* Data Splitting 
* Labeling
* Hyperparameter Tuning
* Model Fine-tuning
*	Model Deployment
*	How to Improve

### Abstract:
The original task was to choose either GPT or BERT model and use these models to do the classification on a dataset of customer complaint comments to put them into different categories and use this for further insights. 
The task is not completely clear since we couldn’t contact the stakeholders enough to know what their different categories are and what is the accuracy of the trained model evaluated upon, which is a crucial part of the project because it helps us get to best possible answer and look up to the main task when we are stuck. But here we are now just trying to solve the problem as good as possible based on everything we know. 
This project is just a showcase of the writers abilities and the output of the project may not be useful considering the lack of clarity of the project’s purpose and also more important than that the limitations of my laptop computational resources and limited time to finish the task. 

### Model Selection:
First, we have to choose which model to use, BERT or GPT?
I chose BERT over GPT for classifying these complaint comments into different categories and here is why:
Our task involves classifying customer complaints, which is typically a sentence-level classification task. BERT is designed to handle sentence-level tasks and has been shown to perform well on a wide range of text classification tasks. 
Another reason for this choice is our data size. Most of the time GPT requires more than our original data set(around 4000) labeled data to work at its best. While BERT requires less than this number and can perform well even with under 1000 labeled data. 
The diversity of our data is not that high meaning that it is typically about complaints of customers about quality of the product or the way it was delivered to them. If it was diverse we might have wanted to choose GPT because it works better with diverse data. But we still can get good answers with BERT since we don’t have high diversity.
These reasons look enough for choosing BERT over GPT in our case but there is another reason which confirms our choice and that is computational resources. GPT requires more computational resources and my normal laptop might struggle using this model for classification.

### Data Preprocessing:
In this step we perform data preprocessing on the dataset to ensure that it is clean, consistent, and in a format that can be used by the model. Here is the R code we used to do this (code is explained in the comments):
```r
# first we read our file. (of course)
df <- read.csv("joined_clean.csv")

# Get the unique values of issue_map to insure there is no error in it 
unique_issue_vals <- unique(df$issue_map)

# We realize that some of the comments are not only in Persian, but also have some English words that are not written by customers. So we try to detect all the comments including at least one English letter. (There were better ways of detecting this but I tried to do it using two external packages but it raised error and I decided to choose this non-professional way!)
english_rows <- df[grepl("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z", df$Comment, ignore.case = TRUE), ]

# We see that there are some of the issue_maps included in the comments so we loop over each unique value and replace it in the Comment column with ""
for (val in unique_issue_vals) {
  df$Comment <- gsub(val, "", df$Comment, ignore.case = TRUE)
}
rm(val)

# Then we check rest of the English words and replace them with "" if they are not needed or replace them with translation of Persian if it was used by customer itself.
df$Comment <- gsub("Damage Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("jira", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Bad-quality Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Wrong Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Biker Behavior", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Fresh QUality", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Near Expiration", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("fragile", "شکستنی", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("D:", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Extra Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("gira", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("error", "ارور", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("c", "سی", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("B", "ب", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("a4", "", df$Comment, ignore.case = TRUE)
df <- df[!(df$X == 108 | df$X == 110 | df$X == 203 | df$X == 912 | df$X == 1311), ]

# Now we check if all the English rows are gone(They are!)
english_rows <- df[grepl("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z", df$Comment, ignore.case = TRUE), ]

# Time to check if the short comments are valid or not. We remove nonsense ones and keep complete comments.
short_df <- df[nchar(df$Comment) < 10, ]
df <- df[!(df$Comment == "" | df$Comment == " " | df$Comment == "  " | df$Comment == " تخم مرغ" | df$X == 63 | df$X == 219 | df$X == 942 | df$X == 1530) ,]
# we realize we don’t need issue_map column so we remove it and create column label since we will need it later.
df <- subset(df, select = -issue_map)
df$label <- ""

# We will use this full data at the end of the project for full labeling prediction of the complete dataset
write.csv(df, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\full_data.csv", row.names=FALSE)

# rest of the code is in the same R file but we will mention it in the next section(data splitting)

```

### Data Splitting:
In this step we split our whole dataset into three separate datasets for train, validation and test. In normal data splitting for BERT model we choose 70% train and 15% for each validation and test. But we don’t want to spend all our time labeling the data, so we pick 10% of our dataset for training and 45% for each of validation and test sets.
```r
# Set the seed for reproducibility
set.seed(657)

# Split the data into training, validation, and test sets
train_indices <- sample(nrow(df), round(0.1*nrow(df)), replace = FALSE)
val_indices <- sample(setdiff(1:nrow(df), train_indices), round(0.45*nrow(df)), replace = FALSE)
test_indices <- setdiff(setdiff(1:nrow(df), train_indices), val_indices)

train_data <- df[train_indices, ]
val_data <- df[val_indices, ]
test_data <- df[test_indices, ]

write.csv(train_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\train_data.csv", row.names=FALSE)
write.csv(val_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\val_data.csv", row.names=FALSE)
write.csv(test_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\test_data.csv", row.names=FALSE)
```

### Labeling:
In this sad step of our project we should read each comment of our train dataset and label them with one category. We decided to devide all of our comments into 4 categories: 
0 -> price |  1 -> product | 2 -> packaging | 3 -> delivery
Tese picked categories are decided by the author and is not the best options for sure. Also we realized how close the packaging and delivery categories seem to be, but considering our lack of time and the fact that the output of this project won’t be used by no one we will stop overthinking this label choosing and will start labeling all of our train data. 

### Hyperparameter Tuning:
In this step we use the following code to choose our hyperparameters. For now we try to tune two parameters: 
'learning_rate': [2e-5, 3e-5, 5e-5] and  'epochs': [3, 4, 5]
The following code was supposed to give us an output letting us know which learning rate and epoch is the best for our dataset, but after couple of hours hearing my laptap sound like a vaccum cleaner I realized this code takes around 24 hours to run so we skip this section with a huge failure and act like nothing has happened. (I could have realized that this code won’t give me the answer in ok time much earlier!)
```python
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import ParameterGrid

# Load the train dataset
train_data = pd.read_csv('val_data.csv')

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
```
### Model Fine-tuning:
In this step we fine-tune the selected model using transfer learning techniques on the training set. This will involve adjusting the model's weights and biases to optimize its performance on the specific task of classifying complaint comments based on issue.
For this purpose we run the following code that provides us with BERT-tokenizer and pythorch_model which will be used in the next step to predict the labels of our dataset. 
```python
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
```

### Model Deployment:
Now that we have fine-tuned our model it is time to test our model. 
We use the following code for this purpose.
```python
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
```
### How to improve:
After reviewing the final result we figure out that there is a lot of space for improvement. The possible reasons that made our results not as good as expected will be mentioned here. 
First we could do better if we had clearer purpose for the task and we had enough time to do so.
Another way to improve this results were to tune the hyperparameters before developing the main fine-tuned model. We couldn’t do this because of the limitations of our computational resources.
One more thing that could lead to a better result was choosing the different categories in better way that they really could be distinguishable for human and also our model. Our choices for different categories(labels) were not precise enough to expect the model to predict it all correctly. 
Besides from hyperparameters tuning that lead to failure due to computational limitations, We could have tried different parameters and check out how our model works. This one was also not possible considering the limitations in computational resources that made most of our running times sooo long.
