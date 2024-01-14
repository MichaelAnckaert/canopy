import json
import os
from copy import deepcopy
from typing import Optional, Any, Dict, cast, Union, Iterable, List

import jsonschema
import openai
from openai.types.chat import ChatCompletionToolParam
from tenacity import retry, stop_after_attempt, retry_if_exception_type

import canopy
from canopy.llm import OpenAILLM
from canopy.llm.models import Function, FunctionParameters, FunctionArrayProperty
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, UserMessage, Query


class AzureOpenAILLM(OpenAILLM):
    """
    Azure OpenAI LLM wrapper built on top of the OpenAI Python client.

    Azure OpenAI services require a valid API key, and an Azure endpoint. You will need
    To set the following environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key.
    - AZURE_OPENAI_ENDPOINT: Your Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`
    
    To use a specific custom model deployment, simply pass the deployment name to the `model_name` argument
    
    Note: If you want to set an OpenAI organization, you would need to set environment variable "OPENAI_ORG_ID". 
          This is different from the environment variable for passing an organization to the parent class (OpenAILLM), which is "OPENAI_ORG".
    """  # noqa: E501
    def __init__(self,
                 model_name: str,
                 *,
                 api_key: Optional[str] = None,
                 api_version: str = "2023-12-01-preview",
                 azure_endpoint: Optional[str] = None,
                 organization: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the Azure OpenAI LLM.

        Args:
            model_name: The name of your custom model deployment on Azure. This is required. 
            api_key: Your Azure OpenAI API key. Defaults to None (uses the "AZURE_OPENAI_API_KEY" environment variable).
            api_version: The Azure OpenAI API version to use. Defaults to "2023-12-01-preview".
            azure_endpoint: The url of your Azure OpenAI service endpoint. Defaults to None (uses the "AZURE_OPENAI_ENDPOINT" environment variable).
            **kwargs: Generation default parameters to use for each request.


        >>> from canopy.llm.azure_openai_llm import AzureOpenAILLM
        >>> from canopy.models.data_models import UserMessage
        >>> llm = AzureOpenAILLM()
        >>> messages = [UserMessage(content="Hello! How are you?")]
        >>> llm.chat_completion(messages)

        """  # noqa: E501
        self.model_name = model_name

        try:
            self._client = openai.AzureOpenAI(
                azure_deployment=model_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                organization=organization,
            )
        except (openai.OpenAIError, ValueError) as e:
            raise RuntimeError(
                "Failed to connect to Azure OpenAI, please make sure that the "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables "
                "are set correctly. "
                f"Underlying Error:\n{self._format_openai_error(e)}"
            )

        self.default_model_params = kwargs

    @property
    def available_models(self):
        raise NotImplementedError(
            "Azure OpenAI LLM does not support listing available models"
        )

    async def achat_completion(
        self,
        messages: Messages,
        *,
        stream: bool = False,
        max_generated_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        raise NotImplementedError()

    async def agenerate_queries(
        self,
        messages: Messages,
        *,
        max_generated_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> List[Query]:
        raise NotImplementedError()

    def _handle_chat_error(self, e):
        if isinstance(e, openai.AuthenticationError):
            raise RuntimeError(
                "Failed to connect to Azure OpenAI, please make sure that the "
                "AZURE_OPENAI_API_KEY environment variable is set correctly. "
                f"Underlying Error:\n{self._format_openai_error(e)}"
            )
        elif isinstance(e, openai.APIConnectionError):
            raise RuntimeError(
                f"Failed to connect to your Azure OpenAI endpoint, please make sure "
                f"that the provided endpoint {os.getenv('AZURE_OPENAI_ENDPOINT')} "
                f"is correct. Underlying Error:\n{self._format_openai_error(e)}"
            )
        else:
            super()._handle_chat_error(e)