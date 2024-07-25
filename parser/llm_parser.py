import os
from typing import List, Tuple, Optional, Dict
import logging
import fitz  # PyMuPDF
import concurrent.futures
from paddleocr import PPStructure
import numpy as np
import cv2
from PIL import Image
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from parser.rect_merge import merge_all
import shapely.geometry as sg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""

DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""

engine = PPStructure(table=False, ocr=False, show_log=True, structure_version="PP-StructureV2")


def _parse_rects(img: Image, page_num: int, page: fitz.Page, output_dir: str) -> List[Dict]:
    """
    Parse drawings in the page and merge adjacent rectangles.
    """
    result = engine(img)
    image_property_list = []
    h, w, _ = img.shape
    res = sorted_layout_boxes(result, w)
    images = page.get_image_info()
    total_image_property_list = []
    for index, info in enumerate(res):
        x1, y1, x2, y2 = info["bbox"]
        element = info["type"]
        total_image_property_list.append({element: ((x1, y1, x2, y2), index)})
    # 计算初始布局中figure的数量
    initial_figure_count = sum(1 for item in total_image_property_list if list(item.keys())[0] == 'figure')

    # 如果get_image_info返回的图片数量大于初始布局中的figure数量，补充缺失的图片
    if len(images) > initial_figure_count:
        for image_info in images[initial_figure_count:]:
            bbox = tuple(int(x) * 2 for x in image_info["bbox"])
            number = image_info["number"] - 1

            # 找到插入位置
            insert_index = next((i for i, item in enumerate(total_image_property_list)
                                 if list(item.values())[0][1] >= number), len(total_image_property_list))

            # 插入新的figure
            total_image_property_list.insert(insert_index, {"figure": (bbox, number)})

            # 更新插入位置之后的所有元素的顺序
            for i in range(insert_index + 1, len(total_image_property_list)):
                key = list(total_image_property_list[i].keys())[0]
                coords, order = total_image_property_list[i][key]
                total_image_property_list[i] = {key: (coords, order + 1)}
    blocks = [(k, v) for block in total_image_property_list for k, v in block.items()]
    merged_blocks = merge_all(blocks, page_height=h, page_width=w)
    for block in merged_blocks:
        # key, value = block.popitem()
        x1, y1, x2, y2 = block[1][0]
        element = block[1][1]
        cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
        name = f"{output_dir}output/page_{page_num + 1}_{element}_{x1}_{y1}_{x2}_{y2}.png"
        cropped_img.save(name)
        image_property_list.append({name: element})
    return total_image_property_list


def _parse_pdf_to_images(pdf_path: str, output_dir: str = './') -> List[List]:
    """
    Parse PDF to images and save to output_dir.
    """
    img_infos = []
    with fitz.open(pdf_path) as pdf:
        # for pg in range(0, pdf.page_count):
        for pg in range(1, 2):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_property_list = _parse_rects(img, pg, page, output_dir)
            img_infos.append(image_property_list)

    return img_infos


def _gpt_parse_images(
        image_infos: List[List[Dict[str, str]]],
        prompt_dict: Optional[Dict] = None,
        output_dir: str = './',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> str:
    """
    Parse images to markdown content.
    """
    from GeneralAgent import Agent

    if isinstance(prompt_dict, dict) and 'prompt' in prompt_dict:
        prompt = prompt_dict['prompt']
        logging.info("prompt is provided, using user prompt.")
    else:
        prompt = DEFAULT_PROMPT
        logging.info("prompt is not provided, using default prompt.")
    if isinstance(prompt_dict, dict) and 'role_prompt' in prompt_dict:
        role_prompt = prompt_dict['role_prompt']
        logging.info("role_prompt is provided, using user prompt.")
    else:
        role_prompt = DEFAULT_ROLE_PROMPT
        logging.info("role_prompt is not provided, using default prompt.")

    def _process_image(page_index: int, img_index: int, image_path: str, image_type: str, page_info: str) -> Tuple[
        int, int, str]:
        logging.info(f'gpt parse image: {image_path}')
        agent = Agent(role=role_prompt, api_key=api_key, base_url=base_url, disable_python_run=True, model=model,
                      **args)

        local_prompt = f"{prompt}\n\nThis image is part of a page with the following information:\n{page_info}\n\nCurrent image type: {image_type}"
        content = agent.run([local_prompt, {'image': image_path}], display=verbose)
        return page_index, img_index, content

    contents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = []
        for page_index, page_images in enumerate(image_infos):
            page_info = "\n".join([f"{list(img.values())[0]}: {list(img.keys())[0]}" for img in page_images])
            for img_index, img in enumerate(page_images):
                image_path = list(img.keys())[0]
                image_type = list(img.values())[0]
                futures.append(
                    executor.submit(_process_image, page_index, img_index, image_path, image_type, page_info))

        for future in concurrent.futures.as_completed(futures):
            page_index, img_index, content = future.result()

            # 在某些情况下大模型还是会输出 ```markdown ```字符串
            if '```markdown' in content:
                content = content.replace('```markdown\n', '')
                last_backticks_pos = content.rfind('```')
                if last_backticks_pos != -1:
                    content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]

            while len(contents) <= page_index:
                contents.append([])
            while len(contents[page_index]) <= img_index:
                contents[page_index].append('')
            contents[page_index][img_index] = content

    # 合并每个页面的内容
    merged_contents = ['\n\n'.join(page_contents) for page_contents in contents]

    output_path = os.path.join(output_dir, 'output.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(merged_contents))

    return '\n\n'.join(merged_contents)


def parse_pdf(
        pdf_path: str,
        output_dir: str = './',
        prompt: Optional[Dict] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> Tuple[str, List[str]]:
    """
    Parse a PDF file to a markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    content = _gpt_parse_images(
        image_infos=image_infos,
        output_dir=output_dir,
        prompt_dict=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        verbose=verbose,
        gpt_worker=gpt_worker,
        **args
    )

    all_rect_images = []
    # remove all rect images
    if not verbose:
        for page_image, rect_images in image_infos:
            if os.path.exists(page_image):
                os.remove(page_image)
            all_rect_images.extend(rect_images)
    return content, all_rect_images


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(override=True)

    prompt = {
        "prompt": DEFAULT_PROMPT,
        "role_prompt": DEFAULT_ROLE_PROMPT
    }
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    content, image_paths = parse_pdf(
        pdf_path="/Users/chenwenhong/Downloads/NewDocuments/gptpdf/examples/attention_is_all_you_need.pdf",
        output_dir="../",
        api_key=api_key,
        base_url=base_url,
        model="azure_gpt-4o",
        prompt=prompt,
        verbose=True,
        gpt_worker=4
    )
    print(content)
