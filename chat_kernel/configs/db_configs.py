#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/28 8:59
"""
import typing as t

KB_ROOT_PATH: str = str("tests")  # 暂时使用测试路径

CHUNK_SIZE: int = 750
"""知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)"""

OVERLAP_SIZE: int = 150
"""知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)"""

ZH_TITLE_ENHANCE: bool = False
"""是否开启中文标题加强，以及标题增强的相关配置"""


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
"""
TextSplitter配置项，如果你不明白其中的含义，就不要修改。
source 如果选择tiktoken则使用openai的方法 "huggingface"
"""

TEXT_SPLITTER_NAME: str = "ChineseRecursiveTextSplitter"
"""TEXT_SPLITTER 名称"""

#
# EMBEDDING_KEYWORD_FILE: str = "embedding_keywords.txt"
# """Embedding模型定制词语的词表文件"""

# FAISS 相关设置
FAISS_CACHE_SIZE = 256
FAISS_LOCATION = ""

