import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.training.loss_functions import kl_loss
from torch import Tensor
from transformers import LlamaForCausalLM, T5ForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config

# from .tricks import generate_distribution, generate_distribution2, generate_distribution3, shift_embeddings


class T5ForConditionalGeneration_baseline(T5ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        decoder_head_mask: torch.FloatTensor | None = None,
        cross_attn_head_mask: Tensor | None = None,
        encoder_outputs: Tuple[Tuple[Tensor]] | None = None,
        past_key_values: Tuple[Tuple[Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_train: bool = False,
    ) -> Tuple[torch.FloatTensor] | Seq2SeqLMOutput:
        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


class T5ForConditionalGeneration_shift(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, alpha: float):
        super().__init__(config)
        self.alpha = alpha

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        decoder_head_mask: torch.FloatTensor | None = None,
        cross_attn_head_mask: Tensor | None = None,
        encoder_outputs: Tuple[Tuple[Tensor]] | None = None,
        past_key_values: Tuple[Tuple[Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_train: bool = False,
    ) -> Tuple[torch.FloatTensor] | Seq2SeqLMOutput:
        output = super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        if is_train:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = shift_embeddings(inputs_embeds, self.alpha)
            outputs2 = super().forward(
                None,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                head_mask,
                decoder_head_mask,
                cross_attn_head_mask,
                encoder_outputs,
                past_key_values,
                inputs_embeds,
                decoder_inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            output["loss"] += outputs2["loss"]
            output["loss"] /= 2

        return output


class LlamaForCausalLM_baseline(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        is_train: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            num_logits_to_keep,
        )
        return outputs
