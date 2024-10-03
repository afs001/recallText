#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/9/27 18:50
"""
import sys
from pathlib import Path

from chat_kernel.utils.loader.pdf_loader import OCRPDFLoader

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
from pprint import pprint

test_files = {
    "ocr_test.pdf": str(root_path / "recallText" / "tests" / "samples" / "A_doc_test" / "AF01_test.pdf"),
}


def test_rapidocrpdfloader():
    pdf_path = test_files["ocr_test.pdf"]

    loader = OCRPDFLoader(pdf_path)
    docs = loader.load()
    pprint(docs)
    assert (
            isinstance(docs, list)
            and len(docs) > 0
            and isinstance(docs[0].page_content, str)
    )
    return docs


if __name__ == '__main__':
    docs = test_rapidocrpdfloader()
