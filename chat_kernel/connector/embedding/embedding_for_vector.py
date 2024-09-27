#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:33
"""
import requests
from langchain_core.embeddings import Embeddings

from chat_kernel.configs.model_configs import LOCAL_EMBED_SERVICE_URL


class TextEmbeddings(Embeddings):
    def __init__(self):
        self.model_version = 'local_v20240725'
        self.url = f"http://{LOCAL_EMBED_SERVICE_URL}/embedding"
        self.session = requests.Session()
        super(TextEmbeddings, self).__init__()




