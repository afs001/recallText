#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/30 8:40
"""
from typing import List

from langchain_core.documents import Document
from pprint import pprint

from chat_kernel.utils.splitter import ChineseTextSplitter
from tests.test_pdfloader import test_rapidocrpdfloader


def test_textsplitter(doc_lst: List[Document]):
    text_splitter = ChineseTextSplitter()
    splited_docs = text_splitter.split_documents(doc_lst)
    pprint(splited_docs)


if __name__ == '__main__':
    docs = test_rapidocrpdfloader()
    test_textsplitter(docs)