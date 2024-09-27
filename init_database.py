#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 19:35
@desc: Docs embedding to index db class.
"""
from abc import ABC, abstractmethod


class DocsEmbeddingFlowBase(ABC):

    @abstractmethod
    def do_init_by_docs(self):
        """
        Init database by docs.
        :return:
        """
        pass

    @abstractmethod
    def parse_pdf(self):
        """
        Parse pdf to text.
        :return:
        """
        pass

    @abstractmethod
    def split_text(self):
        """
        Split text to sentences.
        :return:
        """
        pass

    @abstractmethod
    def chunk_text(self):
        """
        Chunk text to small pieces.
        :return:
        """
        pass

    @abstractmethod
    def embedding_text(self):
        """
        Embedding text to vectors.
        :return:
        """
        pass


class DocsEmbeddingFlow(DocsEmbeddingFlowBase):
    def __init__(self):
        pass

    def do_init_by_docs(self):
        pass

    def parse_pdf(self):
        pass

    def split_text(self):
        pass

    def chunk_text(self):
        pass

    def embedding_text(self):
        pass
