import torch 
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from src.loader.model import load_model
from src.loader.checkpoint import load_pretrained_bag
from src.loader.data import load_data # , get_datacollator
import numpy as np
from tqdm import tqdm
from src.misc.utils import to_cuda
from time import time 

@torch.no_grad()
def generation(model, 
               model_name, 
               batch, 
               tokenizer,
               monomial_processor=None,
               encoding='monomial', # 'monomial' or 'standard' or None
               max_length=1024, quantize_fn=None):
    
    if monomial_processor is None:
        max_length += batch['input_ids'].shape[-1]
        
    preds = model.generate(batch['input_ids'], 
                           attention_mask=batch['attention_mask'],
                                max_length=max_length, 
                                num_beams=1,
                                tokenizer=tokenizer,
                                monomial_processor=monomial_processor,
                                do_sample=False)
    if monomial_processor is not None:
        preds = monomial_processor.batch_decode(preds, skip_special_tokens=True)
    else:
        preds = tokenizer.batch_decode(preds.long().cpu().numpy(), skip_special_tokens=True)

    return preds

def support_eq(pred, label):

    if isinstance(pred, str):
        pred, label = pred.split(' '), label.split(' ')
    
    if len(pred) != len(label):
        return False 
    
    return np.all([p == l for p, l in zip(pred, label) if l != '[C]' and l[0] != 'C' ])

def coeffs_eq(pred, labels_for_regression, th=0, modulo=None):
    target = np.array(labels_for_regression[np.isfinite(labels_for_regression)])
    pred = np.array([float(p[1:]) for p in pred.split(' ') if p[0] == 'C'])
    
    if len(pred) != len(target): 
        return False, np.array([])
    
    if modulo is not None: 
        pred = np.array(pred.round() % modulo, dtype=int)
        
    delta = np.abs(target - pred) 
    
    return np.all(delta <= th), delta
    

def polynomial_eq(pred, label, label_for_regression=None, th=0, modulo=None, compute_support_acc=True):
    
    if compute_support_acc:
        support_hit = support_eq(pred, label)
    else:
        support_hit = None
    
    if label_for_regression is not None:
        coeff_hit, _ = coeffs_eq(pred, label_for_regression, th=th, modulo=modulo)
        hit = support_hit and coeff_hit
    else:
        hit = pred == label
        
    return bool(hit), bool(support_hit) if compute_support_acc else None
    
def accuracy_score(preds, labels, labels_for_regression=None, th=0, modulo=None, compute_support_acc=True):
    
    if labels_for_regression is None:
        labels_for_regression = [None] * len(labels)
    
    hits = []
    support_hits = []
    for pred, label, label_rg in zip(preds, labels, labels_for_regression):
        hit, support_hit = polynomial_eq(pred, label, label_for_regression=label_rg, th=th, modulo=modulo, compute_support_acc=compute_support_acc)
        hits.append(hit)
        support_hits.append(support_hit)
    
    acc = np.array(hits, dtype=float).mean()
    support_acc = np.array(support_hits, dtype=float).mean()
    
    return {'acc': acc, 
            'support_acc': support_acc,
            'hits': hits, 
            'support_hits': support_hits}

@torch.no_grad()
def generation_accuracy(model, dataloader, 
                        batch_size=8, 
                        max_length=1024, 
                        tokenizer=None, 
                        monomial_processor=None,
                        th=0, 
                        disable_tqdm=False, 
                        modulo=None, 
                        model_name=None, 
                        quantize_fn=None, 
                        from_checkpoint=False, 
                        compute_support_acc=True, 
                        **kwargs):
    
    # load model    
    if isinstance(model, str):
        bag = load_pretrained_bag(model, from_checkpoint=from_checkpoint)
        model, tokenizer, model_name = bag['model'], bag['tokenizer'], bag['model_name']
        
    if isinstance(dataloader, str):
        assert(tokenizer is not None)
        # dataset = load_data(dataloader, **kwargs)
        # dc = get_datacollator(model_name)(tokenizer, continuous_coefficient=True)
        # dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dc, shuffle=False)
    
    model.cuda().eval()
    hits = []
    support_hits = []
    runtimes = []
    dataloader = tqdm(dataloader, disable=disable_tqdm) if not disable_tqdm else dataloader
    for batch in dataloader:
        batch = to_cuda(batch)

        max_length = min(max_length, batch['labels'].shape[1])
        
        start = time()
        preds = generation(model, model_name, batch, tokenizer, monomial_processor=monomial_processor, max_length=max_length, quantize_fn=quantize_fn)
        end = time()
        runtime = end - start
        
        
        labels = batch['labels']
        labels[labels == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)        
        
        results = accuracy_score(preds, 
                                 targets, 
                                 th=th,
                                 modulo=modulo,
                                 compute_support_acc=compute_support_acc)
                
        hit = 'x' if preds[0] == targets[0] else ' '
        print(f'preds   [{hit}]: {preds[0]}')
        print(f'targets [{hit}]: {targets[0]}')
        # print(f'labels: {labels[0][:5]}')
        print()
        
        hits.extend(results['hits'])
        # hits.extend(list(mses))
        if compute_support_acc:
            support_hits.extend(results['support_hits'])
            
        runtimes.extend([runtime] * len(preds))
        
    acc = np.array(hits, dtype=float).mean()
    
    if compute_support_acc:
        support_acc = np.array(support_hits, dtype=float).mean() if support_hits else 0.0
    
    results = {'acc': float(acc), 
               'support_acc': float(support_acc) if compute_support_acc else None, 
               'hits': hits, 
               'support_hits': support_hits,
               'batch_runtimes': runtimes,
               'runtime_per_batch': float(np.mean(runtimes))}
    
    
    
    return results 

        