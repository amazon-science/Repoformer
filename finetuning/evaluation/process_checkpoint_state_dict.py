from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# please run zero_to_fp32.py first and save to "pytorch_model.bin.original"
in_dir = sys.argv[1]

model = 'bigcode/starcoderbase-1b'

state_dict = torch.load(in_dir + '/' + 'pytorch_model.bin.original')
state_dict_revised = {k.replace('_forward_module.model.', '').replace('module.model.transformer', 'transformer'): v for k, v in state_dict.items()} 
state_dict_revised['lm_head.weight'] = state_dict_revised['transformer.wte.weight']

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.add_tokens(['<cfc_info>', '</cfc_info>'])

model = AutoModelForCausalLM.from_pretrained(model)
try:
    model.load_state_dict(state_dict_revised)
except:
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(state_dict_revised)
    
model.config.save_pretrained(in_dir)
model.save_pretrained(in_dir)
tokenizer.save_pretrained(in_dir)
print('Saved to', in_dir)
