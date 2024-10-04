#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 18:18
"""
import os
from typing import Dict

import faiss
import pandas as pd
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from chat_kernel.configs.db_configs import FAISS_INDEX_SIZE


class FaissClient:
    def __init__(self):
        self.index = faiss.IndexFlatL2(FAISS_INDEX_SIZE)
        self.metadocs = InMemoryDocstore(),

    def load_vector_store(self, embeddings, vb_path, index2doc_id: Dict = {}):
        if is_directory_empty(vb_path):
            return FAISS(
                embedding_function=embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id=index2doc_id,
            )
        else:
            new_vs = FAISS.load_local(vb_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            return new_vs
    def init_in_memory_doc_store(self, meta={}):
        self.metadocs.add(meta)
        return self.metadocs

def is_directory_empty(directory_path):
    # 检查给定路径是否是一个目录
    if not os.path.isdir(directory_path):
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")

    # 列出目录中的所有文件和文件夹
    return len(os.listdir(directory_path)) == 0


faiss_vector_store = FaissClient()






