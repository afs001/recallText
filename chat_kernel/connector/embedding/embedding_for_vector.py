#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:33
"""
from typing import List

import requests
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class TextEmbeddings(Embeddings):
    def __init__(self):
        self.model = None
        super(TextEmbeddings, self).__init__()

    def init_embedding_model(self, model_path: str):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_query(self, text: str) -> List[float]:
        pass

embedding = TextEmbeddings()




