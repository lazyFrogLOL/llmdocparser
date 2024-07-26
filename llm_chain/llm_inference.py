from langchain_core.messages import HumanMessage, SystemMessage
import dotenv
import base64
from PIL import Image
from io import BytesIO
from typing import List, Optional


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def construct_message(system_prompt: str, user_prompt: str, image_data: str) -> List[SystemMessage | HumanMessage]:
    """
    构造单个系统消息和人类消息。

    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param image_data: 图像的base64编码数据
    :return: 包含SystemMessage和HumanMessage的列表
    """
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
        image_data_list: List[str],
        system_prompt: str = "You are a helpful assistant",
        user_prompt: str = "Describe the content in this image",
        max_concurrency: Optional[int] = None
) -> List[str]:
    """
    批量分析多个图像并返回描述。

    :param llm: LLM模型实例
    :param image_data_list: 包含多个图像base64编码数据的列表
    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param max_concurrency: 最大并发数，如果为None则不限制
    :return: LLM模型的响应内容列表
    """
    try:
        # 构造消息批次
        message_batches = [
            construct_message(system_prompt, user_prompt, image_data)
            for image_data in image_data_list
        ]

        # 设置批处理配置
        batch_config = {}
        if max_concurrency is not None:
            batch_config["max_concurrency"] = max_concurrency

        # 调用LLM模型的batch方法
        responses = llm.batch(message_batches, config=batch_config)

        # 提取每个响应的内容
        return [response.content for response in responses]

    except Exception as e:
        return [f"An error occurred: {str(e)}"]

