#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 21:38
"""
from pathlib import Path

from chat_kernel.configs.db_configs import FAISS_LOCATION
from chat_kernel.configs.model_configs import EMBEDDING_MODEL_PATH
from chat_kernel.connector.database.base import KBService
from chat_kernel.connector.database.faiss import faiss_vector_store
from chat_kernel.connector.embedding import embedding_model

root_path = Path(__file__).parent.parent.parent.parent


class FaissKBService(KBService):
    def __init__(self):
        super(FaissKBService, self).__init__()
        self.vb_path = str(root_path) +  FAISS_LOCATION + "faiss_index"
        self.emb_model = embedding_model.init_embedding_model(EMBEDDING_MODEL_PATH)

    def _load_vector_store(self):
        """Faiss 向量库"""
        return faiss_vector_store.load_vector_store(
                embeddings=self.emb_model,
                vb_path=self.vb_path
            )

    def add_document(self, kb_file, **kwargs):
        docs = kb_file.docs2texts(kb_file.file2docs())

        vector_store = self._load_vector_store()

        # 构造文本块和元数据的列表，假设 kb_file.docs2texts 返回的是文本内容
        texts = [doc.page_content for doc in docs]
        metadata = [doc.metadata for doc in docs]

        # 将文本和对应的元数据添加到向量存储中
        vector_store.add_texts(texts, metadatas=metadata)

        if not kwargs.get("not_refresh_vs_cache"):
            vector_store.save_local(self.vb_path)


    def search(self, query, k=3):
        """根据查询检索相似片段"""
        # 执行相似性搜索，返回与查询最相似的前 3 个文本块
        similar_docs = self._load_vector_store().similarity_search(query, k)

        # 将最相似的文本块和它们的路径聚合为一个结果
        aggregated_result = {}
        for i, doc in enumerate(similar_docs):
            content = doc.page_content
            path = doc.metadata["source"]

            aggregated_result.update({
                "content": content,
                "path": path
            })

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
