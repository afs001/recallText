from typing import List, Union, Any
from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.partition.text import partition_text
import os
import fitz  # PyMuPDF
from tqdm import tqdm
import re


class NewOCRPDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load PDF files with OCR support."""

    def __init__(self, file_path: Union[str, List[str]], mode: str = "single", **unstructured_kwargs: Any):
        super(NewOCRPDFLoader, self).__init__(file_path=file_path, mode=mode, **unstructured_kwargs)
        self.file_path = file_path

    def _get_elements(self) -> List:
        def pdf_ocr_txt(filepath):
            doc = fitz.open(filepath)
            all_text = ""

            # 用 tqdm 显示进度条，显示处理进度
            with tqdm(total=doc.page_count, desc=f"Processing {os.path.basename(filepath)}") as pbar:
                for i in range(doc.page_count):
                    page = doc.load_page(i)  # 加载页面
                    blocks = page.get_text("blocks")  # 获取文本块

                    for block in blocks:
                        block_text = block[4].strip()  # 获取块的实际文本

                        # 去除由于换行符导致的句子被打散的问题
                        block_text = re.sub(r'\n(?!(\n|$))', ' ', block_text)  # 将非段落结束的换行符替换为空格

                        # 去除多余的空格
                        block_text = re.sub(r'\s+', ' ', block_text)  # 将多个空格替换为单个空格

                        # 将文本添加到总文本中
                        all_text += block_text

                    pbar.update(1)  # 更新进度条

            # 处理文本内容，确保段落分开
            sentences = re.split(r'(?<=[。！？!?])', all_text)  # 根据标点符号进行分割
            processed_text = '\n'.join([sentence.strip() for sentence in sentences if sentence.strip()])

            return [processed_text]

        if isinstance(self.file_path, list):
            all_elements = []
            for file in self.file_path:
                print(f"Processing file: {file}")
                text_list = pdf_ocr_txt(file)  # 处理文件，获取纯文本
                for text in text_list:
                    elements = partition_text(text=text, **self.unstructured_kwargs)  # 将文本分割为元素列表
                    all_elements.extend(elements)
            return all_elements  # 返回所有元素
        else:
            print(f"Processing file: {self.file_path}")
            text_list = pdf_ocr_txt(self.file_path)  # 处理单个文件，获取纯文本
            all_elements = []
            for text in text_list:
                elements = partition_text(text=text, **self.unstructured_kwargs)  # 将文本分割为元素列表
                all_elements.extend(elements)
            return all_elements  # 返回元素列表

# 使用示例
# loader = OCRPDFLoader(file_path="example.pdf")
# elements = loader._get_elements()
