#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 18:18
"""

import faiss
import pandas as pd
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from chat_kernel.configs.db_configs import FAISS_INDEX_SIZE


class FaissClient:
    def __init__(self):
        self.index2doc_id = {}
        self.index = faiss.IndexFlatL2(FAISS_INDEX_SIZE)

    def load_vector_store(self, embeddings):
        return FAISS(
            embedding_function=embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id=self.index2doc_id,
        )
faiss_vector_store = FaissClient()






