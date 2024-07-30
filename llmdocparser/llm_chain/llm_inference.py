import pandas as pd
from typing import List, Optional
from langchain.schema import SystemMessage, HumanMessage
from PIL import Image
from io import BytesIO
import base64
from langchain_community.callbacks import get_openai_callback


# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
4. 如果图片中包含图表，对图表形成摘要即可，文字按照markdown格式输出。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""

DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def construct_message(system_prompt: str, user_prompt: str, image_data: str) -> List[SystemMessage | HumanMessage]:
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    return [system_message, human_message]


def analyze_images_batch(
        llm,
        df: pd.DataFrame,
        system_prompt: str = DEFAULT_ROLE_PROMPT,
        user_prompt_template: str = DEFAULT_PROMPT + "\n图片的类型是: {type}\n",
        max_concurrency: Optional[int] = None
) -> pd.DataFrame:
    """
    批量分析DataFrame中的图像并返回添加了新列的DataFrame。

    :param llm: LLM模型实例
    :param df: 包含图像路径的DataFrame
    :param system_prompt: 系统提示词
    :param user_prompt_template: 用户提示词模板，包含 {type} 占位符
    :param max_concurrency: 最大并发数，如果为None则不限制
    :return: 添加了新列的DataFrame
    """
    try:
        # 将图像转换为base64并构造消息批次
        message_batches = []
        for _, row in df.iterrows():
            image_data = image_to_base64(row['filepath'])
            user_prompt = user_prompt_template.format(type=row['type'])
            message_batches.append(construct_message(system_prompt, user_prompt, image_data))

        # 设置批处理配置
        batch_config = {}
        if max_concurrency is not None:
            batch_config["max_concurrency"] = max_concurrency

        # 调用LLM模型的batch方法
        with get_openai_callback() as cb:
            responses = llm.batch(message_batches, config=batch_config)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

        # 提取每个响应的内容
        contents = [response.content for response in responses]

        # 将内容添加到DataFrame中
        df['content'] = contents

        return df

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
