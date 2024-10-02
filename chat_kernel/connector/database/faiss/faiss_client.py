#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 18:18
"""

import faiss
import pandas as pd
from langchain_community.vectorstores import FAISS


class FaissClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings

        self.origin_index = None
        self.origin_metadata = None
        self.faiss_client = FAISS(embedding_function=self.embeddings.embed_query)

    def _load_origin_docs(self):
        self.origin_index = faiss.read_index("faiss_index.bin")
        self.origin_metadata = pd.read_csv("metadata.csv")

    def search(self, query):
        pass

    def merge_docs(self, docs):
        pass

    def add_document(self, docs):
        pass

    def delete_documents(self, kb_id, file_ids=None):
        pass
