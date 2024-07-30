import os
from typing import List, Optional
import logging
import pandas as pd
from llmdocparser.parser.layout_parser import parse_pdf_to_images
from llmdocparser.llm_chain.llm_inference import analyze_images_batch
from llmdocparser.llm_chain.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageContentAnalyzer:
    def __init__(self):
        self.llm = None

    def initialize_llm(self, llm_type: str, **kwargs):
        self.llm = LLMFactory.create_llm(llm_type, **kwargs)

    def clip_pdf(self, pdf_path: str, output_dir: str = './') -> List[List]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_infos = parse_pdf_to_images(pdf_path, output_dir=output_dir)
        return image_infos

    def parse_images(self, pdf_path, output_dir):
        image_each_page_infos = self.clip_pdf(pdf_path, output_dir=output_dir)
        df = pd.DataFrame()
        for page_info in image_each_page_infos:
            sub_df = pd.DataFrame(page_info)
            df = pd.concat([df, sub_df])
        return df

    def get_image_content(self, pdf_path: str, output_dir: str, max_concurrency: Optional[int] = None) -> pd.DataFrame:
        if self.llm is None:
            raise ValueError("LLM has not been initialized. Call initialize_llm first.")

        image_info_df = self.parse_images(pdf_path, output_dir)
        result_df = analyze_images_batch(self.llm, image_info_df, max_concurrency=max_concurrency)
        return result_df


def get_image_content(llm_type: str, pdf_path: str, output_dir: str, max_concurrency: Optional[int] = None, **llm_kwargs) -> pd.DataFrame:
    analyzer = ImageContentAnalyzer()
    analyzer.initialize_llm(llm_type, **llm_kwargs)
    return analyzer.get_image_content(pdf_path, output_dir, max_concurrency)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(override=True)

    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_API_BASE')

    content = get_image_content(
        llm_type="azure",
        pdf_path="llmdocparser/example/attention_is_all_you_need.pdf",
        output_dir="output/",
        max_concurrency=5,
        azure_deployment="azure-gpt-4o",
        azure_endpoint=base_url,
        api_key=api_key,
        api_version="2024-02-01"
    )
    print(content)
