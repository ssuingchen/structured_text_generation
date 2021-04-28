#!/usr/bin/env python
# coding: utf-8

# Reference: 
# 1. https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb
# 2. https://huggingface.co/transformers/task_summary.html#text-generation
# 3. https://huggingface.co/transformers/main_classes/pipelines.html
# 4. https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel
# 5. https://huggingface.co/gpt2?text=A+long+time+ago%2C+

# Three types of baseline from pre-trained GPT-2

# In[1]:


# !pip install transformers
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)


# Greedy Search

# In[ ]:


answer_list = ['Place the chicken, butter, soup, and onion']
while True:
    answer = generator(answer_list[-1], return_full_text = False, max_length=30)
    answer_list.append(answer[0]['generated_text'])
    if len(answer_list) >= 10:
        break
recipe_instruction = "".join(answer_list).replace(". ", ".\n")

print("Output:\n" + 30 * '-')
print(recipe_instruction)


# Beam Search + 2-gram

# In[ ]:


answer_list = ['Place the chicken, butter, soup, and onion']
while True:
    answer = generator(answer_list[-1], return_full_text = False, max_length=30, num_beams=5, no_repeat_ngram_size=2)
    answer_list.append(answer[0]['generated_text'])
    if len(answer_list) >= 10:
        break
recipe_instruction = "".join(answer_list).replace(". ", ".\n")

print("Output:\n" + 30 * '-')
print(recipe_instruction)


# Top-K

# In[ ]:


answer_list = ['Place the chicken, butter, soup, and onion']
while True:
    answer = generator(answer_list[-1], return_full_text = False, do_sample=True, max_length=30, top_k=30)
    answer_list.append(answer[0]['generated_text'])
    if len(answer_list) >= 10:
        break
recipe_instruction = "".join(answer_list).replace(". ", ".\n")

print("Output:\n" + 30 * '-')
print(recipe_instruction)

