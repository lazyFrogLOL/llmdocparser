import os
from typing import List, Tuple, Optional, Dict
import logging
import concurrent.futures
from layout_parser import parse_pdf_to_images
from GeneralAgent import Agent

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


def get_prompt(prompt_dict: Optional[Dict] = None, prompt_key: str = 'prompt', default_prompt: str = DEFAULT_PROMPT) -> str:
    """获取提示词"""
    if isinstance(prompt_dict, dict) and prompt_key in prompt_dict:
        logging.info(f"{prompt_key} is provided, using user prompt.")
        return prompt_dict[prompt_key]
    logging.info(f"{prompt_key} is not provided, using default prompt.")
    return default_prompt


def create_agent(role_prompt: str, api_key: Optional[str], base_url: Optional[str], model: str, **args) -> Agent:
    """创建Agent实例"""
    return Agent(role=role_prompt, api_key=api_key, base_url=base_url, disable_python_run=True, model=model, **args)


def process_image(
        agent: Agent,
        image_path: str,
        image_type: str,
        page_info: str,
        prompt: str,
        verbose: bool,
        page_index: int,
        img_index: int
) -> Tuple[int, int, str]:
    """处理单个图像"""
    logging.info(f'gpt parse image: {image_path}')
    local_prompt = (
        f"{prompt}\n\n"
        f"Current image type: {image_type}"
    )
    content = agent.run([local_prompt, {'image': image_path}], display=verbose)
    return page_index, img_index, clean_content(content)


def clean_content(content: str) -> str:
    """清理内容中的markdown标记"""
    if '```markdown' in content:
        content = content.replace('```markdown\n', '')
        last_backticks_pos = content.rfind('```')
        if last_backticks_pos != -1:
            content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]
    return content


def process_images_concurrently(image_infos: List[List[Dict[str, str]]], agent: Agent, prompt: str, verbose: bool, gpt_worker: int) -> List[List[str]]:
    """并发处理所有图像"""
    contents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = []
        for page_index, page_images in enumerate(image_infos):
            page_info = "\n".join([f"{list(img.values())[0]}: {list(img.keys())[0]}" for img in page_images])
            for img_index, img in enumerate(page_images):
                image_path = list(img.keys())[0]
                image_type = list(img.values())[0]
                futures.append(
                    executor.submit(process_image, agent, image_path, image_type, page_info, prompt, verbose, page_index, img_index)
                )

        for future in concurrent.futures.as_completed(futures):
            page_index, img_index, content = future.result()
            while len(contents) <= page_index:
                contents.append([])
            while len(contents[page_index]) <= img_index:
                contents[page_index].append('')
            contents[page_index][img_index] = content

    return contents


def save_content(content: str, output_dir: str) -> None:
    """保存内容到文件"""
    output_path = os.path.join(output_dir, 'output.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def gpt_parse_images(
        image_infos: List[List[Dict[str, str]]],
        prompt_dict: Optional[Dict] = None,
        output_dir: str = './',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> List:
    """Parse images to markdown content."""
    prompt = get_prompt(prompt_dict, 'prompt', DEFAULT_PROMPT)
    role_prompt = get_prompt(prompt_dict, 'role_prompt', DEFAULT_ROLE_PROMPT)

    agent = create_agent(role_prompt, api_key, base_url, model, **args)
    contents = process_images_concurrently(image_infos, agent, prompt, verbose, gpt_worker)

    return contents


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
) -> str:
    """
    Parse a PDF file to a markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = parse_pdf_to_images(pdf_path, output_dir=output_dir)
    content = gpt_parse_images(
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
    return content


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(override=True)

    prompt = {
        "prompt": DEFAULT_PROMPT,
        "role_prompt": DEFAULT_ROLE_PROMPT
    }
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    content = parse_pdf(
        pdf_path="../example/attention_is_all_you_need.pdf",
        output_dir="../",
        api_key=api_key,
        base_url=base_url,
        model="azure_gpt-4o",
        prompt=prompt,
        verbose=True,
        gpt_worker=1
    )
    print(content)
