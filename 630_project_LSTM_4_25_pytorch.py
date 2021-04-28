#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
from collections import Counter
import torch


# In[2]:


df = pd.read_csv('recipe_list.csv')


# In[10]:


train_df, dev_df, test_df =               np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df)), int(.9*len(df))])


# In[4]:


vocab = set()
for text in df.text:
    words = text.split(' ')
    for word in words:
        vocab.add(word)
vocab_size = len(vocab)
print("Saw %d unique characters in all texts" % len(vocab))


# In[2]:


import transformers
# Load the GPT tokenizer.
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startofrecipe|>', eos_token='<|endofrecipe|>', pad_token='<|pad|>')
# add special tokens for title, ingredients and instruction seperator
special_tokens_dict = {'additional_special_tokens': ['<|startofingre|>', '<|startofinstruc|>']}
# check the number of special tokens
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')


# In[7]:


from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []

        for txt in txt_list:

            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx][:-1]),
            torch.tensor(self.input_ids[idx][1:]),
        )


# In[3]:


from torch import nn

class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


# In[8]:


def train(dataset, dev_dataset, model, device, batch_size, max_epochs, sequence_length, checkpoint_name, training_stats
):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(max_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, max_epochs))
        print('Training...')
        losses = []
        total_train_loss = 0
        model.train()

        state_h, state_c = model.init_state(sequence_length)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()
            
            batch_loss = loss.item()
            total_train_loss += batch_loss
            losses.append(batch_loss)
            
            if batch % 1000 == 0 and not batch == 0:
                print({ 'epoch': epoch+1, 'batch': batch, 'loss': loss.item() })
        

            loss.backward()
            optimizer.step()
            
            if batch % 5000 == 0:
                torch.save(model, checkpoint_name)
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(dataloader)       

        # Calculate perplexity.
        losses = torch.tensor(losses)
        train_perplexity = math.exp(torch.mean(losses))
        
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Perplexity: {0:.2f}".format(train_perplexity))        
        # ========================================
        #               Validation
        # ========================================
        print("")
        print("Running Validation...")
        model.eval()

        losses = []
        total_eval_loss = 0
        nb_eval_steps = 0
        
        state_h, state_c = model.init_state(sequence_length)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        # Evaluate data for one epoch
        for batch, (x, y) in enumerate(dev_dataloader):
        
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():        

                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

            batch_loss = loss.item()
            losses.append(batch_loss)
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(dev_dataloader)

        # Calculate perplexity.
        losses = torch.tensor(losses)
        val_perplexity = math.exp(torch.mean(losses))

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation perplexity: {0:.2f}".format(val_perplexity))        

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Perplexity': train_perplexity,
                'Valid. Perplexity': val_perplexity,
            }
        )
        
    print("")
    print("Training complete!")


# Start creating datasets and training

# In[ ]:


train_dataset = GPT2Dataset(train_df['text'], tokenizer, max_length=200)
dev_dataset = GPT2Dataset(dev_df['text'], tokenizer, max_length=200)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(dev_dataset)))

model = Model(len(tokenizer))
model.to(torch.device('cuda'))

training_stats = []
train(train_dataset, dev_dataset, model, device=torch.device('cuda'), batch_size=2, max_epochs=5, sequence_length=199, checkpoint_name='LSTM1', training_stats=training_stats)


# Load pre-trained model

# In[4]:


# Model class must be defined somewhere
model = torch.load('LSTM1')
model.eval()


# In[5]:


def evaluate(dataset,model, device, batch_size, sequence_length):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
     
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")
    model.eval()

    losses = []
    total_eval_loss = 0
    nb_eval_steps = 0

    state_h, state_c = model.init_state(sequence_length)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    # Evaluate data for one epoch
    for batch, (x, y) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():        

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

        batch_loss = loss.item()
        losses.append(batch_loss)
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(dataloader)

    # Calculate perplexity.
    losses = torch.tensor(losses)
    val_perplexity = math.exp(torch.mean(losses))

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation perplexity: {0:.2f}".format(val_perplexity))        
    print("")
    print("Evaluation complete!")


# Evaluate model with test data

# In[11]:


#prepare test dataset
test_dataset = GPT2Dataset(test_df['text'], tokenizer, max_length=200)
device = torch.device('cuda')
model.to(device)
evaluate(test_dataset, model, device, batch_size = 2, sequence_length = 199)


# generate recipes with inital words

# In[52]:


initial_words = []
for text in df['text'][:50]:
    initial_words.append(text.split(" ")[1])


# Generator with no_repeat_ngram function implemented

# In[92]:


def predict(model, device, tokenizer, text, next_words=200):
    
    model.eval()

    text = '<|startofrecipe|> ' + text
    input_word_ids = tokenizer(text)['input_ids']
    state_h, state_c = model.init_state(len(input_word_ids))
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    generated_word_ids = list(input_word_ids)
    
    #record unique input_word_ids
    no_repeat_bigram = {}
    no_repeat_bigram[str(input_word_ids)] = 1
    
    for i in range(0, next_words):
        
        x = torch.tensor([input_word_ids], device=device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().to('cpu')
        p = np.array(p)
        
        # get the non repeated trigram before adding to generated_word_ids
        while True:
            next_word_id = np.random.choice(len(last_word_logits), p=p)
            
            # make sure the trigram of input_word_ids and next_word_id is unique
            new_input_word_ids = [input_word_ids[0], input_word_ids[1], next_word_id]
            
            # if dict value get back 0, means there is no such pair of bigram in dict, 
            # add new pair of ids to input_word_ids and dict
            if no_repeat_bigram.get(str(new_input_word_ids), 0) == 0:
                no_repeat_bigram[str(new_input_word_ids)] = 1
                input_word_ids.append(next_word_id)
                input_word_ids = input_word_ids[1:]
                break
            else:
                continue
            
        generated_word_ids.append(next_word_id)
        
        if input_word_ids[-1] == tokenizer.eos_token_id:
            print('here')
            break
        
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_word_ids))


# In[95]:


predict(model, device, tokenizer, text= "Chicken", next_words=500)


# In[51]:


generated_recipes = []
for initial_word in initial_words:
    predict(model, device, tokenizer, text= "Chicken", next_words=200)
    


# In[ ]:




