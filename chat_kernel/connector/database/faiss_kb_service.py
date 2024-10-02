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

from chat_kernel.configs.model_configs import EMBEDDING_MODEL_PATH
from chat_kernel.connector.database.base import KBService
from chat_kernel.connector.embedding.embedding_for_vector import embedding
from init_database import KnowledgeFile, get_kb_path

class FaissKBService(KBService):
    def __init__(self):
        super(FaissKBService, self).__init__()
        self.index = None  # FAISS索引
        self.documents = []  # 文档元数据
        self.embeddings = []  # 存储向量
        self.embedding_dim = 0  # 向量维度
        self.is_initialized = False  # 标志索引是否初始化
        self.emb_model_path = EMBEDDING_MODEL_PATH

    def _load_vector(self):
        """嵌入模型"""
        return embedding.init_embedding_model(model_path=self.emb_model_path)

    def add_document(self, kb_file: KnowledgeFile, **kwargs) -> List[Dict]:
        docs = kb_file.docs2texts(kb_file.file2docs())

        texts = [x.page_content for x in docs]
        metas = [x.metadata for x in docs]
        # 对分块后的文本进行嵌入
        embeddings = self._load_vector().embed_documents(texts)
        ids = self._load_vector().add_embeddings(
            text_embeddings=zip(texts, embeddings), metadatas=metas
        )

        if not kwargs.get("not_refresh_vs_cache"):
            self._load_vector().save_local(self.vs_path)

        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def search(self, query, k=5):
        """根据查询检索相似片段"""
        pass

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


if __name__ == "__main__":
    faissService = FaissKBService()
    faissService.add_document(KnowledgeFile("README.md", "test"))
    faissService.remove_document(KnowledgeFile("README.md", "test"))
    print(faissService.search("如何启动api服务"))
