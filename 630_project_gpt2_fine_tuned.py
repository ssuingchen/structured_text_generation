#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')


# Project Data Preprocessing

# In[4]:


import pandas as pd
import json


# In[3]:


# load and also preprocess the raw data
def load_preprocess_raw_data(raw_data):
    '''
    take raw recipe data and preprocess it, 
    return a list of recipe instances with special tokens

    parameter: raw data

    return: recipe instance list

    '''
    with open(raw_data, 'r') as f:
        raw_dict = json.load(f)
    f.close()

    raw_list = []
    for recipe in raw_dict.values():
        # try/except will filter out recipes that don't have title, ingredients or instructions
        try:
            title = recipe['title'].replace("ADVERTISEMENT", "")
            ingredient_list = recipe['ingredients']
            ingredients = ""
            for ingredient in ingredient_list:
                ingredient = ingredient.replace("ADVERTISEMENT", "")
                if ingredient != "":
                    ingredients += ingredient + ", "
            instructions = recipe['instructions'].replace("ADVERTISEMENT", "")
            recipe_instance = '<|startofrecipe|>'+title+'<|startofingre|>'+ingredients+'<|startofinstruc|>'+instructions+'<|endofrecipe|>'
            if len(recipe_instance) <= 2000:
                raw_list.append(recipe_instance)

        except:
            continue
    return raw_list


# In[4]:


# create text list for dataset
recipe_one_list = load_preprocess_raw_data("recipes_raw_nosource_ar.json")
recipe_two_list = load_preprocess_raw_data("recipes_raw_nosource_epi.json")
recipe_three_list = load_preprocess_raw_data("recipes_raw_nosource_fn.json")
recipe_list = recipe_one_list + recipe_two_list + recipe_three_list
train_list, test_list = np.split(recipe_list, [int(.8*len(recipe_list))])
print('Number of train data: ', len(train_list))
print('Number of test data: ', len(test_list))


# Tokenizer

# In[2]:


# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startofrecipe|>', eos_token='<|endofrecipe|>', pad_token='<|pad|>')
# add special tokens for title, ingredients and instruction seperator
special_tokens_dict = {'additional_special_tokens': ['<|startofingre|>', '<|startofinstruc|>']}
# check the number of special tokens
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')


# Pytorch Dataset

# In[6]:


class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


# In[7]:


dataset = GPT2Dataset(train_list, tokenizer, max_length=200)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# In[11]:


batch_size = 2


# In[12]:


# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# Finetune

# In[3]:


# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[22]:


# some parameters I cooked up that work reasonably well

epochs = 3
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 1000
# I save the model every 5000 step
save_every = 5000
# save the model to this file name
save_file = 'trial_2'


# In[23]:


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )


# In[24]:


# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
print('Total number of steps: ', total_steps)
# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


# In[25]:


training_stats = []
print("Currently using device type: ", device)

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    losses = []

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask =b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss
        losses.append(batch_loss)

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:
            print('Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.'.format(step, len(train_dataloader), batch_loss))

        loss.backward()

        optimizer.step()

        scheduler.step()

        if step % save_every == 0:
            model.save_pretrained(save_file)

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
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

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        losses.append(batch_loss)
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Calculate perplexity.
    losses = torch.tensor(losses)
    val_perplexity = math.exp(torch.mean(losses))

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation perplexity: {0:.2f}".format(val_perplexity))        

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Perplexity': train_perplexity,
            'Valid. Perplexity': val_perplexity,
        }
    )

print("")
print("Training complete!")


# In[26]:


model.save_pretrained(save_file)


# Evaluate the test data

# In[28]:


# prepare datasets for dev_list and test_list
test_dataset = GPT2Dataset(test_list, tokenizer, max_length=768)


# In[29]:


# load the datasets
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# In[30]:


def evaluate_model(model, dataloaded):
    model = model.to(device)
    model.eval()

    losses = []
    perplexity = []
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in dataloaded:

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():        

            outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                            attention_mask = b_masks,
                            labels=b_labels)

            loss = outputs[0]  

        batch_loss = loss.item()
        losses.append(batch_loss)
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(dataloaded)

    # Calculate perplexity.
    losses = torch.tensor(losses)
    val_perplexity = math.exp(torch.mean(losses))
    perplexity.append(val_perplexity)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation perplexity: {0:.2f}".format(val_perplexity))
    return avg_val_loss, val_perplexity


# In[31]:


print('Testing...')
test_loss, test_perplexity = evaluate_model(model, test_dataloader)
test_eval_df = pd.DataFrame(columns = ["test_loss", "test_perplexity"])
test_eval_df['test_loss'] = test_loss
test_eval_df['test_perplexity'] = test_perplexity
test_eval_df.to_csv("test_eval.csv")


# Load pre-trained model

# In[19]:


model = GPT2LMHeadModel.from_pretrained("trial_2", config=configuration)


# create initial words for generating recipes

# In[5]:


df = pd.read_csv('recipe_list.csv')
df.head()


# In[14]:


initial_words = []
for text in df['text']:
    initial_words.append(text.split(" ")[1])
    


# In[17]:


initial_words


# In[21]:


generated_recipes = []
for i in range(len(initial_words[:50])):    
    input_ids = tokenizer(initial_words[i], return_tensors='pt').input_ids
    model.to(input_ids.device)
    sample_outputs = model.generate(
                                        input_ids,
                                        num_beams=5, 
                                        no_repeat_ngram_size=2, 
                                        max_length = 400,
                                        num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id
                                    )
    generated_recipes.append(tokenizer.decode(sample_outputs[0]))


# In[22]:


generated_recipes[:5]


# Create CSV files

# In[23]:


generated_recipes_df = pd.DataFrame(columns = ["text"])
generated_recipes_df['text'] = generated_recipes
generated_recipes_df.to_csv("generated_recipes.csv")


# In[27]:


training_eval_df = pd.DataFrame(columns = ["epoch", "training loss", "validation loss", "train perplexity", "validation perplexity"])
train_loss = []
train_perp = []
valid_loss = []
valid_perp = []
for epoch in training_stats:
    train_loss.append(epoch['Training Loss'])
    train_perp.append(epoch['Training Perplexity'])
    valid_loss.append(epoch['Valid. Loss'])
    valid_perp.append(epoch['Valid. Perplexity'])

training_eval_df['epoch'] = [x for x in range(len(training_stats))]
training_eval_df['training loss'] = train_loss
training_eval_df['validation loss'] = valid_loss
training_eval_df['train perplexity'] = train_perp
training_eval_df['validation perplexity'] = valid_perp

training_eval_df.to_csv("training evaluation.csv")
    


# In[ ]:




