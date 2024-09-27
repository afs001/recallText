#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:42
@desc: Query for related texts
"""
from abc import ABC, abstractmethod


class QueryFlowBase(ABC):

    @abstractmethod
    def do_query(self):
        """
        Query for related texts
        :return:
        """
        pass

    @abstractmethod
    def split_query(self):
        """
        Split the query text
        :return:
        """
        pass

    @abstractmethod
    def search_similar_texts(self):
        """
        Search for the texts that are similar to the query text
        :return:
        """
        pass

    @abstractmethod
    def rerank_texts(self):
        """
        Re-rank the texts based on the query text
        :return:
        """
        pass


class QueryFlow(QueryFlowBase):
    def __init__(self):
        pass

    def do_query(self):
        pass

    def split_query(self):
        pass

    def search_similar_texts(self):
        pass

    def rerank_texts(self):
        pass


if __name__ == '__main__':
    pass
