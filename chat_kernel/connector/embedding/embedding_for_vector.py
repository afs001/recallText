#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:33
"""
from typing import List

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from chat_kernel.configs.model_configs import EMBEDDING_MODEL_PATH


class TextEmbeddings:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda', "No GPU available!"

    def init_embedding_model(self, model_path: str = EMBEDDING_MODEL_PATH):
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs = {'device': self.device}
        )

embedding = TextEmbeddings()




