from typing import Optional, Tuple, Union, List

import torch 
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.tokenization_utils import PreTrainedTokenizer

from src.loader.data_format.processors.base import BaseProcessor

from transformers.models.bart.modeling_bart import (
    BartEncoder, 
    BartDecoder, 
    BartModel, 
    BartForConditionalGeneration, 
    BartConfig, 
    shift_tokens_right
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput
)

from .encodings.monomial import MonomialEmbedding

class CustomBartEncoder(BartEncoder):
    '''
    Custom BartEncoder with additional embedding processing layers.

    Args:
        config (BartConfig): Configuration for the encoder
        embed_tokens (Optional[nn.Embedding]): Token embedding layer.
            If None, uses default BartEncoder token embedding
        post_token_embedding (Optional[nn.Module]): Additional processing layer after token embedding.
            Must implement forward(inputs_embeds, attention_mask=None)
        more_position_embedding (Optional[nn.Module]): Additional position embedding layer.
            Applied after post_token_embedding
    '''
    def __init__(self, 
                 config: BartConfig, 
                 embed_tokens: Optional[nn.Embedding] = None,
                 monomial_embedding: Optional[MonomialEmbedding] = None,
                 post_token_embeddings: Optional[nn.ModuleDict] = {},
                 more_position_embeddings: Optional[nn.ModuleDict] = {}
                 ):
        
        super().__init__(config, embed_tokens=embed_tokens)

        self.post_token_embeddings = post_token_embeddings
        self.more_position_embeddings = more_position_embeddings 
        self.monomial_embedding = monomial_embedding
        
        post_token_embeddings_keys = list(post_token_embeddings.keys()) if post_token_embeddings is not None else []
        more_position_embeddings_keys = list(more_position_embeddings.keys()) if more_position_embeddings is not None else []
        self.additional_kwargs = ['token_types', 'monomial_types'] + post_token_embeddings_keys + more_position_embeddings_keys

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutput]:
        
        if self.monomial_embedding is not None:
            inputs_embeds = self.monomial_embedding(input_ids)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        
        for key in self.post_token_embeddings:
            embedding_fn = self.post_token_embeddings[key]
            position_input = kwargs[key]
            token_embeds = embedding_fn(position_input)
            inputs_embeds += token_embeds
            
        for key in self.more_position_embeddings:
            embedding_fn = self.more_position_embeddings[key]
            position_input = kwargs[key]
            position_embeds = embedding_fn(position_input)
            inputs_embeds += position_embeds
        
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['attention_mask'] = attention_mask

        for key in self.additional_kwargs:
            if key in kwargs:
                kwargs.pop(key)

        return super().forward(**kwargs)
        
        
class CustomBartDecoder(BartDecoder):
    '''
    Custom BartDecoder with additional embedding processing layers.

    Args:
        config (BartConfig): Configuration for the decoder
        embed_tokens (Optional[nn.Embedding]): Token embedding layer.
            If None, uses default BartDecoder token embedding
        post_token_embedding (Optional[nn.Module]): Additional processing layer after token embedding.
            Must implement forward(inputs_embeds, attention_mask=None)
        more_position_embedding (Optional[nn.Module]): Additional position embedding layer.
            Applied after post_token_embedding
    '''
    def __init__(self, 
                 config: BartConfig, 
                 embed_tokens: Optional[nn.Embedding] = None,
                 monomial_embedding: Optional[MonomialEmbedding] = None,
                 post_token_embeddings: Optional[nn.ModuleDict] = {},
                 more_position_embeddings: Optional[nn.ModuleDict] = {}
                 ):
        
        super().__init__(config, embed_tokens=embed_tokens)

        self.post_token_embeddings = post_token_embeddings
        self.more_position_embeddings = more_position_embeddings 
        self.monomial_embedding = monomial_embedding


        post_token_embeddings_keys = list(post_token_embeddings.keys()) if post_token_embeddings is not None else []
        more_position_embeddings_keys = list(more_position_embeddings.keys()) if more_position_embeddings is not None else []   
        self.additional_kwargs = ['token_types', 'monomial_types'] + post_token_embeddings_keys + more_position_embeddings_keys
        
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        if self.monomial_embedding is not None:
            inputs_embeds = self.monomial_embedding(input_ids)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        
        for key in self.post_token_embeddings:
            embedding_fn = self.post_token_embeddings[key]
            position_input = kwargs[key]
            token_embeds = embedding_fn(position_input)
            inputs_embeds += token_embeds[:, :-1]  
            
        for key in self.more_position_embeddings:
            embedding_fn = self.more_position_embeddings[key]
            position_input = kwargs[key]
            position_embeds = embedding_fn(position_input)
            inputs_embeds += position_embeds[:, :-1] 
        
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['attention_mask'] = attention_mask

        for key in self.additional_kwargs:
            if key in kwargs:
                kwargs.pop(key)
        
        return super().forward(**kwargs)
        
        
class CustomBartModel(BartModel):
    '''
    Custom BartModel with additional embedding processing layers.

    Args:
        config (BartConfig): Configuration for the model
        post_token_embedding (Optional[nn.Module]): Additional processing layer after token embedding.
            Must implement forward(inputs_embeds, attention_mask=None)
        more_position_embedding (Optional[nn.Module]): Additional position embedding layer.
            Applied after post_token_embedding

    Note:
        The additional embedding layers are applied independently to encoder and decoder
    '''
    def __init__(self,
                 config: BartConfig,
                 post_token_embeddings: Optional[nn.ModuleDict] = {},
                 more_position_embeddings: Optional[nn.ModuleDict] = {},
                 monomial_embedding: Optional[MonomialEmbedding] = None
                 ):
        super().__init__(config)
        
        self.monomial_embedding = monomial_embedding
        
        # Create custom encoder and decoder
        self.encoder = CustomBartEncoder(
            config,
            embed_tokens=self.shared,
            post_token_embeddings=post_token_embeddings,
            more_position_embeddings=more_position_embeddings,
            monomial_embedding=monomial_embedding
        )

        self.decoder = CustomBartDecoder(
            config,
            embed_tokens=self.shared,
            post_token_embeddings=post_token_embeddings,
            more_position_embeddings=more_position_embeddings,
            monomial_embedding=monomial_embedding
        )

        # Initialize weights and apply final processing
        self.post_init()
            
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            #additional
            input_token_types: Optional[torch.LongTensor] = None,
            decoder_token_types: Optional[torch.LongTensor] = None,
            input_monomial_types: Optional[torch.LongTensor] = None,
            decoder_monomial_types: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, Seq2SeqModelOutput]:
            # different to other models, Bart automatically creates decoder_input_ids from
            # input_ids if no decoder_input_ids are provided
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                        "passed, `input_ids` cannot be `None`. Please pass either "
                        "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                    )

                decoder_input_ids = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    #additional
                    token_types=input_token_types,
                    monomial_types=input_monomial_types,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                #additional
                token_types=decoder_token_types,
                monomial_types=decoder_monomial_types,
            )

            if not return_dict:
                return decoder_outputs + encoder_outputs

            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )



class CustomBartForConditionalGeneration(BartForConditionalGeneration):
    '''
    Custom BartForConditionalGeneration with additional embedding processing layers.

    Args:
        config (BartConfig): Configuration for the model
        post_token_embedding (Optional[nn.Module]): Additional processing layer after token embedding.
            Must implement forward(inputs_embeds, attention_mask=None)
        more_position_embedding (Optional[nn.Module]): Additional position embedding layer.
            Applied after post_token_embedding

    Note:
        The additional embedding layers are applied independently to encoder and decoder
    '''
    def __init__(self,
                 config: BartConfig,
                 post_token_embeddings: Optional[nn.ModuleDict] = {},
                 more_position_embeddings: Optional[nn.ModuleDict] = {},
                 monomial_embedding: Optional[MonomialEmbedding] = None
                 ):
        super().__init__(config)

        # Create custom BART model
        self.model = CustomBartModel(
            config,
            post_token_embeddings=post_token_embeddings,
            more_position_embeddings=more_position_embeddings,
            monomial_embedding=monomial_embedding
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        #additional
        input_token_types: Optional[torch.LongTensor] = None,
        decoder_token_types: Optional[torch.LongTensor] = None,
        input_monomial_types: Optional[torch.LongTensor] = None,
        decoder_monomial_types: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                raise ValueError("The `use_cache` argument is changed to `False` since `labels` is provided.")
            
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # additional
            input_token_types=input_token_types,
            decoder_token_types=decoder_token_types,
            input_monomial_types=input_monomial_types,
            decoder_monomial_types=decoder_monomial_types,
        )

        if self.model.monomial_embedding is not None:
            expanded_states = self.model.monomial_embedding(
                            outputs.last_hidden_state, 
                            mode='decode'
                        )
            lm_logits = self.lm_head(expanded_states)
        else:
            lm_logits = self.lm_head(outputs[0])
        
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        monomial_processor: Optional[BaseProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 100,
        **kwargs
    ) -> torch.LongTensor:
        """
        Greedy generation from infix representation. 
        """
        
        if monomial_processor is None:
            assert(input_ids.dim() == 2)  # standard input id sequence
            # Explicitly filter kwargs for the superclass generate method
            expected_super_kwargs = {}
            if 'num_beams' in kwargs:
                expected_super_kwargs['num_beams'] = kwargs['num_beams']
            if 'do_sample' in kwargs:
                expected_super_kwargs['do_sample'] = kwargs['do_sample']
            
            return super().generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=max_length, 
                **expected_super_kwargs
            )
        
        batch_size = input_ids.shape[0]
        num_tokens_per_monomial = input_ids.shape[2]
        
        # Calculate encoder outputs (used as cache)
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Initialize decoder input (with <s> token)
        # decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device) * self.config.decoder_start_token_id
        decoder_input_ids = torch.zeros((batch_size, max_length+1, num_tokens_per_monomial), dtype=torch.long, device=input_ids.device)
        decoder_input_ids[:, 0, -1] = monomial_processor.special_to_id['<s>']
        # decoder_input_ids[:, 1:, -1] = monomial_processor.special_to_id['</s>']
        
        is_end = torch.zeros((batch_size), dtype=torch.bool, device=input_ids.device)
        
        # Generation loop
        for i in range(max_length // num_tokens_per_monomial + 1):

            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids[:, :i+2, :],
                encoder_outputs=encoder_outputs,
                use_cache=True,
                return_dict=True
            )
            
            # 2. Predict next token
            next_token_logits = outputs.logits[~is_end, -num_tokens_per_monomial:, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = tokenizer.batch_decode(next_tokens)
            next_tokens = monomial_processor.generation_helper(next_tokens)

            next_tokens = torch.tensor(next_tokens, dtype=torch.long, device=input_ids.device)

            decoder_input_ids[~is_end, i+1] = next_tokens
            
            is_end[~is_end] = is_end[~is_end] | (next_tokens[:, -1] == monomial_processor.special_to_id['</s>'])
            
            if is_end.all():
                break
        
        return decoder_input_ids