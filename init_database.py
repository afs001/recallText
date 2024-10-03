#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 19:35
@desc: Docs embedding to index db class.
"""
import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, MarkdownHeaderTextSplitter

from chat_kernel.configs.db_configs import *
from chat_kernel.connector.database import faissService
from chat_kernel.utils.loader.pdf_loader import OCRPDFLoader
from chat_kernel.utils.splitter import (
    zh_title_enhance as func_zh_title_enhance,
)


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "A_doc_test")


def get_file_path(knowledge_base_name: str, doc_name: str):
    doc_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (doc_path / doc_name).resolve()
    if str(file_path).startswith(str(doc_path)):
        return str(file_path)


@lru_cache()  # 用于实现最近最少使用（Least Recently Used, LRU）缓存机制
def make_text_splitter(splitter_name, chunk_size, chunk_overlap):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if (
                splitter_name == "MarkdownHeaderTextSplitter"
        ):  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name][
                "headers_to_split_on"
            ]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module("chat_kernel.utils.splitter")
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if (
                    text_splitter_dict[splitter_name]["source"] == "tiktoken"
            ):  # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            elif (
                    text_splitter_dict[splitter_name]["source"] == "huggingface"
            ):  # 从huggingface加载
                if (
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"]
                        == "gpt2"
                ):
                    from langchain.text_splitter import CharacterTextSplitter
                    from transformers import GPT2TokenizerFast

                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module("langchain.text_splitter")
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter


class KnowledgeFile:
    def __init__(self,
                 filename: str,
                 knowledge_base_name: str,
                 ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool = False):
        """
        将文件加载为Document对象
        :param refresh: 如果为True，则重新加载文件
        :return: 返回加载后的Document对象
        """
        if self.docs is None or refresh:
            try:
                loader = OCRPDFLoader(self.filepath)
                if isinstance(loader, TextLoader):
                    loader.encoding = "utf8"
                self.docs = loader.load()
            except Exception as e:
                raise RuntimeError(f"Failed to load file {self.filepath}: {e}")
        return self.docs

    def file2text(self,
                  zh_title_enhance: bool = ZH_TITLE_ENHANCE,
                  refresh: bool = False,
                  chunk_size: int = CHUNK_SIZE,
                  chunk_overlap: int = OVERLAP_SIZE,
                  text_splitter: TextSplitter = None,
                  ):
        """
        将文件加载为文本
        :param zh_title_enhance: 是否增强中文标题
        :param refresh: 是否重新加载
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠大小
        :param text_splitter: 分块器
        :return: 返回加载后的文本
        """
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        return self.splited_docs

    def docs2texts(self,
                   docs: List[Document] = None,
                   zh_title_enhance: bool = ZH_TITLE_ENHANCE,
                   refresh: bool = False,
                   chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = OVERLAP_SIZE,
                   text_splitter: TextSplitter = None,
                   ):
        """
        Split the docs to texts.
        :param docs:
        :return:
        """
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


if __name__ == '__main__':
    from pprint import pprint

    kb_file = KnowledgeFile(
        filename="AF01_test.pdf",
        knowledge_base_name="samples"
    )
    faissService.add_document(kb_file)
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    # docs = kb_file.file2docs()
    # # pprint(docs[-1])
    # texts = kb_file.docs2texts(docs)
    # for text in texts:
    #     print(text)
