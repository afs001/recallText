#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/28 8:59
"""
import typing as t

KB_ROOT_PATH: str = str("source")  # 暂时使用测试路径

"""知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)"""
CHUNK_SIZE: int = 750


"""知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)"""
OVERLAP_SIZE: int = 150


"""是否开启中文标题加强，以及标题增强的相关配置"""
ZH_TITLE_ENHANCE: bool = False


"""
TextSplitter配置项，如果你不明白其中的含义，就不要修改。
source 如果选择tiktoken则使用openai的方法 "huggingface"
"""
text_splitter_dict: t.Dict[str, t.Dict[str, t.Any]] = {
    "ChineseRecursiveTextSplitter": {
        "source": "",
        "tokenizer_name_or_path": "",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on": [
            ("#", "head1"),
            ("##", "head2"),
            ("###", "head3"),
            ("####", "head4"),
        ]
    },
}


"""TEXT_SPLITTER 名称"""
TEXT_SPLITTER_NAME: str = "ChineseRecursiveTextSplitter"

"""TEXT_SPLITTER路径"""
TEXT_SPLITTER_PATH = "A_document"


#
# EMBEDDING_KEYWORD_FILE: str = "embedding_keywords.txt"
# """Embedding模型定制词语的词表文件"""

# FAISS 相关设置
FAISS_INDEX_SIZE = 1024 # "bge-large-zh-v1.5"
FAISS_LOCATION = "/chat_kernel/connector/database/faiss/"

