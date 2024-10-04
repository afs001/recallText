#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 16:42
@desc: Query for related texts
"""
import pandas as pd

from chat_kernel.connector.database import faissService
from chat_kernel.connector.embedding import embedding_model

emb_model = embedding_model.init_embedding_model()

if __name__ == '__main__':

    # Read the CSV file into a DataFrame
    df = pd.read_csv("./source/A_list/A_question.csv")
    contents = []
    paths = []
    embeddings = []
    for value in df.iloc[:, 1]:
        res = faissService.search(value, k=1)
        contents.append(res.get("content", ""))
        embedding = emb_model.embed_query(res.get("content", ""))
        embedding_str = ','.join(map(str, embedding))
        embeddings.append(embedding_str)

        paths.append(res.get("path", ""))

        # Add the new columns to the DataFrame
    df["answer"] = contents
    # df["Path"] = paths
    df["embedding"] = embeddings

    # Save the updated DataFrame to a new CSV file
    df.to_csv("./source/A_list/A_question_answer.csv", index=False)

