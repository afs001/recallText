#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 18:41
"""
from __future__ import annotations


from abc import ABC, abstractmethod

from langchain.vectorstores import VectorStore


class BaseRetrieverService(ABC):
    def __init__(self, **kwargs):
        self.do_init(**kwargs)

    @abstractmethod
    def do_init(self, **kwargs):
        pass

    @abstractmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        pass

    @abstractmethod
    def get_relevant_documents(self, query: str):
        pass
