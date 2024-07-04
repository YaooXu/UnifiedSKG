#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch
import torch
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ..base_models.modeling_t5 import T5ForConditionalGeneration

class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
            args.bert.location,
        )
        
        # added to be compatible with higher version transformers
        self.main_input_name = 'input_ids'
        
        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels, dist_mat=None):        
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            encoder_position_bias=dist_mat
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, dist_mat, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            encoder_position_bias=dist_mat,
            **kwargs,
        )

        return generated_ids
