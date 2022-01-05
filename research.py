import pandas as pd
import torch
import time
import json
import torch.nn.functional as F
import numpy as np
from progressbar import progressbar as pb

'''
# BERT
from transformers import AutoTokenizer, AutoModel, BertModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
one_line = [1]*(len(inputs[0])-2) + [0.5, 0]
attention_mask = torch.tensor([one_line]*8)



outputs = model(inputs, attention_mask=attention_mask)
'''

#Longformer
import torch
from transformers import LongformerModel, LongformerTokenizer

model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to global attention to be deactivated for all tokens
global_attention_mask[:, [1, 4, 21,]] = 1  # Set global attention to random tokens for the sake of this example
                                    # Usually, set global attention based on the task. For example,
                                    # classification: the <s> token
                                    # QA: question tokens
                                    # LM: potentially on the beginning of sentences and paragraphs
mask = torch.zeros((12, 1, 4096, 1, 513), requires_grad=True)
min_mask = torch.ones((12, 1, 4096, 1, 513),) * -10000
min_mask_abs = -min_mask.sum()

# (layer_num, batch_size, seq_len, num_heads, 513)
attention_weights = mask.expand(12, 1, 4096, 12, 513)

outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, attention_weights=attention_weights)
sequence_output = outputs.last_hidden_state
pooled_output = outputs.pooler_output