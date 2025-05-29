from torch.utils.data import Dataset

class DictDatasetBase(Dataset):
    """Base class for datasets"""
    def __init__(self, input_texts, target_texts, **kwargs):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """Dataset-specific initialization"""
        raise NotImplementedError

class DataCollatorBase:
    """Base class for data collators"""
    def __init__(self, **kwargs):
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """Collator-specific initialization"""
        raise NotImplementedError
    
    def __call__(self, batch):
        raise NotImplementedError