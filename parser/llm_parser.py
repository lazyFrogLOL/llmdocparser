import os
from typing import List
import logging
from layout_parser import parse_pdf_to_images
import pandas as pd
from llm_chain.llm_inference import analyze_images_batch
from llm_chain.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clip_pdf(
        pdf_path: str,
        output_dir: str = './',

) -> List[List]:
    """
    Parse a PDF file to a markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = parse_pdf_to_images(pdf_path, output_dir=output_dir)
    return image_infos


def parse_images(pdf_path, output_dir):
    image_each_page_infos = clip_pdf(pdf_path, output_dir=output_dir)
    df = pd.DataFrame()
    for page_info in image_each_page_infos:
        sub_df = pd.DataFrame(page_info)
        df = pd.concat([df, sub_df])
    return df


def get_image_content(llm, pdf_path, output_dir, max_concurrency: int):
    image_info_df = parse_images(pdf_path, output_dir)
    result_df = analyze_images_batch(llm, image_info_df, max_concurrency=max_concurrency)
    return result_df


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(override=True)

    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_API_BASE')
    llm = LLMFactory.create_llm(
        llm_type="azure",
        azure_deployment="azure-gpt-4o",
        azure_endpoint=base_url,
        api_key=api_key
    )
    content = get_image_content(
        llm,
        pdf_path="../example/attention_is_all_you_need.pdf",
        output_dir="../output/",
        max_concurrency=5
    )
    print(content)