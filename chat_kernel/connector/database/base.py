#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 21:37
"""
from chat_kernel.configs.model_configs import VECTOR_SEARCH_TOP_K

from abc import ABC, abstractmethod


class KBService(ABC):
    @abstractmethod
    def add_document(self, document):
        """添加文档到知识库"""
        pass

    @abstractmethod
    def search(self, query, k=VECTOR_SEARCH_TOP_K):
        """根据查询检索相似片段"""
        pass

    @abstractmethod
    def update_document(self, document_id, new_document):
        """更新指定文档"""
        pass

    @abstractmethod
    def remove_document(self, document_id):
        """移除指定文档"""
        pass

    @abstractmethod
    def clear(self):
        """清空知识库"""
        pass
