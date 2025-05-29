from typing import List
from enum import Enum
import string

class ProcessTarget(Enum):
    INPUT = "input"
    TARGET = "target"
    BOTH = "both"

class BaseProcessor:
    """Base class for processors"""
    def __init__(self, target: ProcessTarget = ProcessTarget.BOTH):
        self.target = target

    def __call__(self, texts: List[str], is_target: bool = False) -> List[str]:
        if (self.target == ProcessTarget.BOTH or 
            (self.target == ProcessTarget.INPUT and not is_target) or
            (self.target == ProcessTarget.TARGET and is_target)):
            return self.process(texts)
        return texts

    def process(self, texts: List[str]) -> List[str]:
        raise NotImplementedError

class ProcessorChain(BaseProcessor):
    """Class for chaining processors"""
    def __init__(self, processors: List[BaseProcessor]):
        self.processors = processors

    def __call__(self, texts: List[str], is_target: bool = False) -> List[str]:
        for processor in self.processors:
            texts = processor(texts, is_target)
        return texts

class CoefficientPostfixProcessor(BaseProcessor):
    """Processor to convert coefficients to postfix notation"""
    def process(self, texts: List[str]) -> List[str]:
        return [self._flip_CE(text) for text in texts]
    
    def _flip_CE(self, text: str) -> str:
        polys = text.split(' [SEP] ')
        polys = [self._flip_CE_single(poly) for poly in polys]
        return ' [SEP] '.join(polys)
    
    def _flip_CE_single(self, poly: str) -> str:
        terms = poly.split('+')
        terms_ = []
        for term in terms:
            items = term.strip().split()
            if items:  # Check for empty string
                term_ = items[1:] + [items[0]]
                terms_.append(' '.join(term_))
        return ' + '.join(terms_)

class ExtractLastProcessor(BaseProcessor):
    """Processor to extract the last polynomial"""
    def __init__(self, separator: str = ' [SEP] '):
        super().__init__(target=ProcessTarget.TARGET)  # Apply only to target
        self.separator = separator
    def process(self, texts: List[str], separator: str = None) -> List[str]:
        separator = self.separator if separator is None else separator
        return [text.split(separator)[-1] for text in texts]
    
class ExtractLeadingTermProcessor(BaseProcessor):
    """Processor to extract the first term"""
    def __init__(self, separator: str = ' [SEP] '):
        super().__init__(target=ProcessTarget.TARGET)  # Apply only to target
        self.separator = separator
        
    def process(self, texts: List[str], separator: str = None) -> List[str]:
        separator = self.separator if separator is None else separator
        
        lt_texts = []
        for text in texts:
            polys = text.split(separator)
            leading_terms = [poly.split(' + ')[0] for poly in polys]
            lt_texts.append(separator.join(leading_terms))
        
        return lt_texts

class MultiCoefficientPostfixProcessor(BaseProcessor):
    """Processor to convert coefficients to postfix notation"""
    def process(self, texts: List[str], 
                coeff_token_size: int = 2) -> List[str]:
        if coeff_token_size < 2: 
            return texts
        
        self.coeff_token_size = coeff_token_size
        self.tags = list(string.ascii_lowercase)[:coeff_token_size]
        return [self._expand_CE(text) for text in texts]
    
    def _expand_CE(self, text: str) -> str:
        polys = text.split(' [SEP] ')
        polys = [self._expand_CE_single(poly) for poly in polys]
        return ' [SEP] '.join(polys)
    
    def _expand_CE_single(self, poly: str) -> str:
        terms = poly.split('+')
        terms_ = []
        for term in terms:
            items = term.strip().split()
            if items:  
                term_ = [items[0]+tag for tag in self.tags] + items[1:]
                terms_.append(' '.join(term_))
        return ' + '.join(terms_)

class CoefficientMaskProcessor(BaseProcessor):
    """Processor to replace coefficient tokens (C*) with [C]"""
    def __init__(self, target: ProcessTarget = ProcessTarget.TARGET):
        super().__init__(target=target)
        self.mask_token = "[C]"
    
    def process(self, texts: List[str]) -> List[str]:
        return [self._mask_coefficients(text) for text in texts]
    
    def _mask_coefficients(self, text: str) -> str:
        polys = text.split(' [SEP] ')
        polys = [self._mask_coefficients_single(poly) for poly in polys]
        return ' [SEP] '.join(polys)
    
    def _mask_coefficients_single(self, poly: str) -> str:
        terms = poly.split('+')
        terms_ = []
        for term in terms:
            items = term.strip().split()
            if items:  # Check for empty string
                # Replace first term (coefficient) with mask token if it starts with 'C'
                items[0] = self.mask_token if items[0].startswith('C') else items[0]
                terms_.append(' '.join(items))
        return ' + '.join(terms_)



'''
#usage
# Combination of processors
processors = ProcessorChain([
    CoefficientPostfixProcessor(),  # Apply to both input and output
    ExtractLastProcessor()          # Apply only to output
])

# Create dataset
dataset = StandardDataset(
    input_texts=input_texts,
    target_texts=target_texts,
    processor=processors
)
    
'''
