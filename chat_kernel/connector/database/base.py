#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 21:37
"""
import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from chat_kernel.configs.model_configs import VECTOR_SEARCH_TOP_K
from init_database import get_kb_path, get_doc_path, KnowledgeFile

from abc import ABC, abstractmethod


class KBService(ABC):
    def __init__(self, knowledge: KnowledgeFile):
        self.kb_file = knowledge


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
