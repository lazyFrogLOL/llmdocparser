import os
import re
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures
from paddleocr import PPStructure,save_structure_res
from paddle.utils import try_import
import numpy as np
import cv2
from PIL import Image
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
DEFAULT_RECT_PROMPT = """图片中用红色框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""
DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""

engine = PPStructure(table=False, ocr=False, show_log=True, structure_version="PP-StructureV2")


def _parse_rects(img, page_num, output_dir) -> List[Dict]:
    """
    Parse drawings in the page and merge adjacent rectangles.
    """
    result = engine(img)
    image_property_list = []
    h, w, _ = img.shape
    res = sorted_layout_boxes(result, w)
    for info in res:
        x1, y1, x2, y2 = info["bbox"]
        element = info["type"]
        cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
        name = f"{output_dir}output/page_{page_num + 1}_{element}_{x1}_{y1}_{x2}_{y2}.png"
        cropped_img.save(name)
        image_property_list.append({name:element})
    return image_property_list


def _parse_pdf_to_images(pdf_path: str, output_dir: str = './') -> List[List]:
    """
    Parse PDF to images and save to output_dir.
    """
    img_infos = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_property_list = _parse_rects(img, pg, output_dir)
            img_infos.append(image_property_list)

    return img_infos


