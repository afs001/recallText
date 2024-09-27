#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:33
"""
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class TextEmbeddings(Embeddings):
    def __init__(self):
        self.model_path = ""
        super(TextEmbeddings, self).__init__()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_query(self, text: str) -> List[float]:
        pass




