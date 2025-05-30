import torch
import yaml 
import os 
import argparse
import re 
from transformers import PretrainedConfig
from transformers import PreTrainedTokenizerFast
from src.loader.model import load_model
from safetensors import safe_open

def load_config(save_path, from_checkpoint=False):
    if from_checkpoint:
        cpid = get_checkpoint_id(save_path)
        config_path = os.path.join(save_path, f'checkpoint-{cpid}/config.json')
    else:
        config_path = os.path.join(save_path, 'config.json')
    config = PretrainedConfig.from_json_file(config_path)

    config.regression_weight = 0.1
    return config

def load_pretrained_model(config, save_path, from_checkpoint=False, device_id=0, cuda=True):
    
    model_config = load_config(save_path, from_checkpoint=from_checkpoint)
    
    if from_checkpoint:
        cpid = get_checkpoint_id(save_path)
        checkpoint_path = os.path.join(save_path, f'checkpoint-{cpid}')
    else:
        checkpoint_path = save_path
    
    tokenizer = load_tokenizer(save_path)
    
    if config.model == 'bart':
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(os.path.join(checkpoint_path, f'model.safetensors'), config=model_config, use_safetensors=True)    
        # model = model.to_bettertransformer()
    
    if config.model == 'bart+':
        from loader.models._custom_bart import BartForConditionalGenerationPlus
        model = BartForConditionalGenerationPlus.from_pretrained(os.path.join(checkpoint_path, f'model.safetensors'), config=model_config, use_safetensors=True)
        # model = model.to_bettertransformer()  # not impelemnted
        
    if config.model == 'custom_bart':
        from src.loader.models.custom_bart import CustomBartForConditionalGeneration
        # model = BartForGeneration.from_pretrained(os.path.join(checkpoint_path, f'model.safetensors'), config=model_config, use_safetensors=True)
        config.token_expander = 'mlp1'
        # Create an empty CustomBartForConditionalGeneration model using load_model
        from src.loader.model import load_model
        model = load_model(tokenizer=tokenizer, params=config, device="cuda" if cuda else "cpu")
        
        # Load weights from the checkpoint file
        if cuda:
            model_state_dict = torch.load(os.path.join(checkpoint_path, f'pytorch_model.bin'))
        else:
            model_state_dict = torch.load(os.path.join(checkpoint_path, f'pytorch_model.bin'), map_location='cpu')
            
        model.load_state_dict(model_state_dict)
    
    if cuda: model.cuda()
    
    model.eval()
    
    return model, tokenizer

def get_checkpoint_id(save_dir):
    cpt_file = [f for f in os.listdir(save_dir) if 'checkpoint' in f][0]
    cpid = int(re.search(r'checkpoint-(\d+)', cpt_file).group(1))
    return cpid 

def load_tokenizer(save_dir):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(save_dir, f'tokenizer'))
    return tokenizer

def load_pretrained_bag(save_path, from_checkpoint=False, cuda=True):
    # load config in json format
    with open(os.path.join(save_path, 'training_config.json'), 'r') as f:
        config = yaml.safe_load(f)
        config = argparse.Namespace(**config)
    
    model, tokenizer = load_pretrained_model(config, save_path, from_checkpoint=from_checkpoint, cuda=cuda)
    return {'model': model, 'tokenizer': tokenizer, 'config': config, 'model_name': config.model}
    
    