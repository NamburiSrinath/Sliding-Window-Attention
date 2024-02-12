import types
from typing import Callable, Optional
from modeling_llama import _make_causal_mask as replaced_mask
import transformers.models.llama.modeling_llama as llama_module
from functools import partial
import time
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Seq2SeqTrainer
)
from peft import LoraConfig
from trl import SFTTrainer

def testing_eg(checkpoint):
    prompt = "Who is Leonardo Da Vinci?"
    # TODO: Currently I think the testing pipeline will use the vanilla-att! modify it! 
    pipe = pipeline(task="text-generation", model=checkpoint, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


def see_logs():
    from tensorboard import notebook
    import logging

    log_dir = "results/runs"
    notebook.start("--logdir {} --port 4000".format(log_dir))

    # logging.set_verbosity(logging.CRITICAL)

def swa_attention():
    #### REPLACE THE VANILLA ATTENTION WITH SWA before finetuning!
    # useful to pass extra parameters
    wrapper_function = partial(replaced_mask, window_size=3)
    # Override the original function
    llama_module._make_causal_mask = wrapper_function

prefix = "Summarize the following bill. Focus your summary on the most important aspects of the bill. You do not have to summarize everything. Particularly focus on questions related to appropriation and the effects and impacts of the bill. However, you do not need to go into complex details, it is acceptable to provide ranges. Use active verbs to describe the bill, like 'amends' or 'changes'. Do not use ambivalent verbs like 'proposes' or 'suggests.'"

def preprocess_function(examples):
   inputs = [prefix + doc for doc in examples['text']]
   model_inputs = tokenizer(inputs, truncation=True, padding=True,
                              max_length=1024)
   
   labels = tokenizer(text_target=examples["summary"], padding=True,
                        max_length=512, truncation=True)

   model_inputs["labels"] = labels['input_ids']
   return model_inputs

# Using the chat version helps to debug easily with test input!!
checkpoint = '/hdd4/zoo/llama2/llama2-7b-chat-hf'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Finetune on Billsum dataset!
billsum_dataset = "billsum"
no_epochs = 5
new_model = f"llama-2-7b-finetune-swa-billsum-{no_epochs}"
dataset = load_dataset(billsum_dataset, split="ca_test")

billsum = dataset.train_test_split(test_size=0.2)
tokenized_billsum = billsum.map(preprocess_function, batched=True)

compute_dtype = getattr(torch, "float16")

# Initialize quantization configuration! 
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, #load the model in 4 bits
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = LlamaForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=quant_config, # load the model using quantization config
    device_map='cuda:0'
)
# for training, we shouldn't use KV-cache, as it won't make sense
model.config.use_cache = False
# no tensor parallalism
model.config.pretraining_tp = 1

# Replace vanilla attention with SWA before finetuning the model!
swa_attention()

# LoRA parameters for finetuning!
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# initialize training parameters in the args!
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=int(no_epochs),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# finetune using SFTTrainer class!
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_billsum['train'],
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# train the model! Comment this one line is enough!
trainer.train()

# Save the models
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# see_logs()
# Logging!!
# testing_eg(checkpoint)
# testing_eg(new_model)
