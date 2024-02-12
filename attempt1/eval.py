'''
On a high level overview
1. Load the model
2. Replace the forward function with the modified attention (SWA)
3. Use it to generate the text instead of vanilla attention!

Low level details:
Every model has generate, forward functions that we can use. The model.generate() will eventually call forward() function which access these attention weights. In addition to that, the generate also takes a configuration file 
(https://github.com/huggingface/transformers/blob/3e93dd295b5343557a83bc07b0b2ea64c926f9b4/src/transformers/generation/utils.py#L1452)depending on which it can update/process the generation accordingly!
'''
from transformers import (
   LlamaForCausalLM, 
   LlamaTokenizerFast
)
import torch
from modeling_llama import _make_causal_mask as replaced_mask
import transformers.models.llama.modeling_llama as llama_module
from functools import partial
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import random 

random.seed(42)
torch.manual_seed(42)

def swa_attention():
    """
    This function replaces vanilla attention with SWA, we call it before finetuning
    """
    print("SWA Attention done!")
    # useful to pass extra parameters
    wrapper_function = partial(replaced_mask, window_size=3)
    # Override the original function
    llama_module._make_causal_mask = wrapper_function

checkpoint = str(sys.argv[1])
swa_attention_flag = int(sys.argv[2])
do_sample_flag = int(sys.argv[3])

tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
batch_size = 1

# Evaluate the model with Billsum dataset!
billsum = load_dataset("billsum", split="ca_test")

# testing with a smaller subset!
billsum = billsum.train_test_split(test_size=0.9, seed=42)
print("Data loaded!")

# prefix = f"""
#               Write a summary of the following text delimited by triple backticks.
#               Return your response which covers the key points of the text.
#            """

prefix = "Summarize the following bill. Focus your summary on the most important aspects of the bill. You do not have to summarize everything. Particularly focus on questions related to appropriation and the effects and impacts of the bill. However, you do not need to go into complex details, it is acceptable to provide ranges. Use active verbs to describe the bill, like 'amends' or 'changes'. Do not use ambivalent verbs like 'proposes' or 'suggests.'"

def preprocess_function(examples):
   # inputs = [prefix + doc + "``` SUMMARY: " for doc in examples['text']]
   inputs = [prefix + doc for doc in examples['text']]
   model_inputs = tokenizer(inputs, padding=True, truncation=False, 
                              max_length=1024, return_tensors='pt')
   model_inputs["labels"] = examples["summary"]
   return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)
rouge = evaluate.load('rouge')
inputs = tokenized_billsum['train']['input_ids']
labels = tokenized_billsum['train']['labels']

train_dataloader = DataLoader(tokenized_billsum['train'], 
                           batch_size=batch_size, 
                           shuffle=False,
                           drop_last=True)

print("Inputs created!")

model = LlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
device = 'cuda:0'

if swa_attention_flag == 1:
   swa_attention()

if do_sample_flag == 1:
   do_sample = True
if do_sample_flag == 0:
   do_sample = False

print(f"Do Sample is {do_sample}")
model.to(device)
model.eval()

rouge1, rouge2, rougeL, rougeLsum = 0, 0, 0, 0
a = 0
start_time = time.time()
for batch in train_dataloader:
   print(f"Batch---: {a}")
   with torch.no_grad():
      # input_ids = torch.tensor(batch['input_ids']).to(device)
      input_ids = torch.from_numpy(np.asarray(batch['input_ids'])).to(device).reshape(batch_size, -1)

      attn_masks = torch.from_numpy(np.asarray(batch['attention_mask'])).to(device).reshape(batch_size, -1)
      # print(attn_masks)
      # labels = torch.from_numpy(np.asarray(batch['labels'])).to(device)
      
      # input_ids = batch['input_ids'].to(device)
      labels = batch['labels']
      # print(input_ids.shape)
      # print(labels)
      output = model.generate(input_ids, 
                              max_new_tokens=300,  
                              num_beams=3,
                              do_sample=do_sample_flag,
                              top_k=100,
                              temperature=10.0,
                              no_repeat_ngram_size=5,
                              attention_mask=attn_masks)
      
      predictions = tokenizer.batch_decode(output[:, input_ids.shape[1]:], 
                                             skip_special_tokens=True)

      
      # predictions = tokenizer.decode(output, skip_special_tokens=True)
      # actual_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
      # print(actual_inputs)

      # Aggregating the results!
      results = rouge.compute(predictions=predictions, references=labels)

      rouge1 += results['rouge1']
      rouge2 += results['rouge2']
      rougeL += results['rougeL']
      rougeLsum += results['rougeLsum']

      # Printing one example in every batch to observe!
      print("Input")
      print(batch['text'][0])
      print("-"*100)

      print("Predicted Output")
      print(predictions[0])
      print("-"*100)

      print("Actual Output")
      print(labels[0])
      print("-"*100)

      print(results)
      a += 1 

print(f"Rouge 1, 2, L and Lsum are: {rouge1}, {rouge2}, {rougeL}, {rougeLsum}")
print(f"No of examples processed are {a*batch_size}")
end_time = time.time()
print(f"Total time: {end_time - start_time}")
