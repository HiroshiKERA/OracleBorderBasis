from typing import List 
from src.loader.data_format.processors.base import BaseProcessor, ProcessTarget

class ExtractKLeadingTermsProcessor(BaseProcessor):
    """Processor to extract the first term"""
    def __init__(self, k=-1, separator: str = ' [SEP] ', supersparator: str = ' [BIGSEP] '):
        super().__init__(target=ProcessTarget.INPUT)  # apply only to input
        self.separator = separator
        self.supersparator = supersparator
        self.k = k
        
    def process(self, texts: List[str], separator: str = None) -> List[str]:
        separator = self.separator if separator is None else separator
        
        lt_texts = []
        for text in texts:
            '''
            text = 'L [BIGSEP] V'
            '''
            L, V = text.split(self.supersparator)
            
            if self.k > 0:
                polys = V.split(self.separator)
                
                extract_leading_terms = lambda k, p_text: ' + '.join(p_text.split(' + ')[:k])
                leading_components = [extract_leading_terms(self.k, poly) for poly in polys]
                
                V = f' {self.separator} '.join(leading_components)
            
            new_text = f'{L} {self.supersparator} {V}'
            lt_texts.append(new_text)


        return lt_texts