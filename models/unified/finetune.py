#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from torch import nn
import torch

from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from models.base_models.modeling_t5_ori import T5ForConditionalGeneration


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
            args.bert.location,
        )
        self.config = self.pretrain_model.config
        self.main_input_name = 'input_ids'
        
        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels, encoder_position_bias=None):
        attention_mask = torch.ne(encoder_position_bias, -128)
        
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            encoder_position_bias=encoder_position_bias
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, encoder_position_bias=None, **kwargs):
        attention_mask = torch.ne(encoder_position_bias, -128)

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            encoder_position_bias=encoder_position_bias, 
            **kwargs,
        )

        return generated_ids
