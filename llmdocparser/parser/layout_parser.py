from typing import List, Dict, Tuple
from PIL import Image
import fitz
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from llmdocparser.parser.rect_merge import merge_all
from paddleocr import PPStructure
import numpy as np
import cv2
import os

engine = PPStructure(table=False, ocr=False, show_log=True, structure_version="PP-StructureV2")


def get_initial_layout(img: Image, result: Dict) -> List[Dict]:
    """获取初始布局"""
    h, w, _ = img.shape
    res = sorted_layout_boxes(result, w)
    return [{info["type"]: ((info["bbox"][0], info["bbox"][1], info["bbox"][2], info["bbox"][3]), index)}
            for index, info in enumerate(res)]


def count_initial_figures(layout: List[Dict]) -> int:
    """计算初始布局中figure的数量"""
    return sum(1 for item in layout if list(item.keys())[0] == 'figure')


def supplement_missing_figures(layout: List[Dict], images: List[Dict]) -> List[Dict]:
    """补充缺失的图片"""
    initial_figure_count = count_initial_figures(layout)
    for image_info in images[initial_figure_count:]:
        bbox = tuple(int(x) * 2 for x in image_info["bbox"])
        number = image_info["number"] - 1
        insert_index = next((i for i, item in enumerate(layout)
                             if list(item.values())[0][1] >= number), len(layout))
        layout.insert(insert_index, {"figure": (bbox, number)})
        for i in range(insert_index + 1, len(layout)):
            key = list(layout[i].keys())[0]
            coords, order = layout[i][key]
            layout[i] = {key: (coords, order + 1)}
    return layout


def merge_page_blocks(blocks: List[Tuple[str, Tuple]], page_height: int, page_width: int) -> List[Tuple[str, Tuple]]:
    """合并块"""
    return merge_all(blocks, page_height=page_height, page_width=page_width)


def save_cropped_images(img: Image, blocks: List[Tuple[str, Tuple]], output_dir: str, page_num: int, filename: str, page_height, page_width) -> List[Dict]:
    """保存裁剪后的图像，将裁剪区域往外扩充 20 个像素"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_property_list = []
    for block in blocks:
        x1, y1, x2, y2 = block[1][0]
        element = block[0]
        # 扩充裁剪区域
        x1 = max(0, x1 - 20)  # 防止越界
        y1 = max(0, y1 - 20)
        x2 = min(page_width, x2 + 20)
        y2 = min(page_height, y2 + 20)
        cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
        name = f"{output_dir}/page_{filename}_{page_num + 1}_{element}_{x1}_{y1}_{x2}_{y2}.png"
        cropped_img.save(name)
        image_property_list.append({
            "filepath": name,
            "type": element,
            "page_no": page_num + 1,
            "filename": filename
        })
    return image_property_list


def parse_rects(img: Image, page_num: int, page: fitz.Page, output_dir: str, filename: str) -> List[Dict]:
    """解析页面中的矩形并合并相邻的矩形"""
    result = engine(img)
    initial_layout = get_initial_layout(img, result)
    images = page.get_image_info()
    layout_with_figures = supplement_missing_figures(initial_layout, images)

    blocks = [(k, v) for block in layout_with_figures for k, v in block.items()]
    h, w, _ = img.shape
    merged_blocks = merge_page_blocks(blocks, page_height=h, page_width=w)

    image_property_list = save_cropped_images(
        img,
        merged_blocks,
        output_dir,
        page_num,
        filename,
        page_height=h,
        page_width=w
    )

    return image_property_list


def parse_pdf_to_images(pdf_path: str, output_dir: str = './') -> List[List]:
    """
    Parse PDF to images and save to output_dir.
    """
    img_infos = []
    filename = pdf_path.split("/")[-1].split(".")[0]
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
            image_property_list = parse_rects(img, pg, page, output_dir, filename)
            img_infos.append(image_property_list)

    return img_infos
