#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/10/3 15:03
"""
import os

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from sympy.codegen import Print

embeddings = HuggingFaceEmbeddings(
    model_name="E:/Transformers/bge-large-zh-v1.5",
    model_kwargs = {'device': 'cuda'}
)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

new_vector_store = FAISS.load_local("../chat_kernel/connector/database/faiss/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

num_docs = new_vector_store.index.ntotal
print(num_docs)


results = new_vector_store.similarity_search(
    "网络安全是?",
    k=2,
)
print(results[0])
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

