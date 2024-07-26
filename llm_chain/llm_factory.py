from langchain_openai import AzureChatOpenAI, ChatOpenAI
from typing import Optional, Dict, Any


class LLMFactory:
    @staticmethod
    def create_llm(llm_type: str, **kwargs: Any) -> Any:
        """
        Create and return an LLM instance based on the provided type and parameters.

        :param llm_type: Type of LLM to create ('azure', 'openai', or 'dashscope')
        :param kwargs: Additional parameters for LLM initialization
        :return: An instance of the specified LLM
        """
        if llm_type == 'azure':
            return LLMFactory._create_azure_llm(**kwargs)
        elif llm_type == 'openai':
            return LLMFactory._create_openai_llm(**kwargs)
        elif llm_type == 'dashscope':
            return LLMFactory._create_dashscope_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    @staticmethod
    def _create_azure_llm(**kwargs: Any) -> AzureChatOpenAI:
        required_params = ['azure_deployment', 'azure_endpoint', 'api_key']
        LLMFactory._check_required_params(required_params, kwargs)
        return AzureChatOpenAI(**kwargs)

    @staticmethod
    def _create_openai_llm(**kwargs: Any) -> ChatOpenAI:
        required_params = ['model_name', 'openai_api_key']
        LLMFactory._check_required_params(required_params, kwargs)
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _create_dashscope_llm(**kwargs: Any) -> ChatOpenAI:
        required_params = ['model_name', 'openai_api_key']
        LLMFactory._check_required_params(required_params, kwargs)
        kwargs['base_url'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _check_required_params(required_params: list, provided_params: Dict[str, Any]):
        missing_params = [param for param in required_params if param not in provided_params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")