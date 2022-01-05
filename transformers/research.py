import pandas as pd
import torch
import time
import json
import torch.nn.functional as F
import numpy as np
from progressbar import progressbar as pb

from transformers import AutoTokenizer, AutoModel, BertModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')

outputs = model(inputs, attention_mask=attention_mask)