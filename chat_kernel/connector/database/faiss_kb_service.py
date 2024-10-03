#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 21:38
"""
import os
import shutil
from typing import Dict, List, Tuple

from langchain.docstore.document import Document

from chat_kernel.configs.db_configs import FAISS_LOCATION
from chat_kernel.configs.model_configs import EMBEDDING_MODEL_PATH
from chat_kernel.connector.database.base import KBService
from chat_kernel.connector.database.faiss import faiss_vector_store

class FaissKBService(KBService):
    def __init__(self):
        super(FaissKBService, self).__init__()
        self.vb_path = FAISS_LOCATION + "faiss_index"
        self.emb_model_path = EMBEDDING_MODEL_PATH

    def _load_vector_store(self):
        """Faiss 向量库"""
        return faiss_vector_store.load_vector_store(
            self.emb_model_path
        )

    def add_document(self, kb_file, **kwargs):
        docs = kb_file.docs2texts(kb_file.file2docs())

        self._load_vector_store().add_documents(docs)

        if not kwargs.get("not_refresh_vs_cache"):
            self._load_vector_store().save_local(self.vb_path)


    def search(self, query, k=3):
        """根据查询检索相似片段"""
        # 执行相似性搜索，返回与查询最相似的前 3 个文本块
        similar_docs = self._load_vector_store().similarity_search(query, k)

        # 将最相似的文本块和它们的路径聚合为一个结果
        aggregated_result = ""
        for i, doc in enumerate(similar_docs):
            content = doc.page_content
            path = doc.metadata["path"]
            doc_id = doc.metadata["doc_id"]

            aggregated_result += f"Document {i + 1} (Path: {path}, Doc ID: {doc_id}):\n{content}\n\n"

        # 输出聚合结果
        print("Aggregated Result of Top 3 Documents:")
        print(aggregated_result)
        return aggregated_result

    def update_document(self, document_id, new_document):
        """更新指定文档，需实现索引更新逻辑"""
        # 这需要更复杂的逻辑，通常是删除旧索引然后添加新索引
        raise NotImplementedError("Update functionality is not implemented yet.")

    def remove_document(self, document_id):
        """移除指定文档，需实现索引更新逻辑"""
        # 这也需要复杂的逻辑来处理索引更新
        raise NotImplementedError("Remove functionality is not implemented yet.")

    def clear(self):
        """清空知识库"""
        pass


faissService = FaissKBService()

if __name__ == "__main__":
    pass
