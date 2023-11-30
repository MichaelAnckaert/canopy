import os
from datetime import date, timedelta
import json
import requests
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Union, cast
from canopy.llm.openai import AzureOpenAILLM

from openai.types.chat import ChatCompletionToolParam
from canopy.llm.models import Function, FunctionParameters, FunctionPrimitiveProperty
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query, FunctionMessage, AssistantMessage

IAM_TOKEN = os.getenv("IAM_TOKEN")


def get_calendar():
    start_date = date.today().isoformat()
    end_date = date.today() + timedelta(days=7)
    end_date = end_date.isoformat()
    url = f"https://iam.dewaele.com/api/planner/1?start={start_date}&end={end_date}"

    response = requests.get(url, headers={"Authorization": f"token {IAM_TOKEN}"})
    data = response.json()

    response = f"Dit zijn de afspraken die gepland staan in de agenda van de gebruiker. Als je de afspraken meedeelt aan de gebruiker, groepeer dan de afspraken per dag, met als titel van de sectie datum.\n\n"
    for event in data:
        response += f"* {event['title']} on {event['start']}\n"

    return response


def find_properties(keywords):
    keywords = keywords.replace(" ", "%20")
    url = f"https://iam.dewaele.com/api/search?query={keywords}"

    response = requests.get(url, headers={"Authorization": f"token {IAM_TOKEN}"})
    data = response.json()

    response = f"Dit zijn de panden die voldoen aan de sleutelwoorden die gebruiker opgaf. De URL van het pand moet je als HTML hyperlink tag aan de gebruiker teruggeven.\n\n"

    for result in data:
        if result["_type"] == "property":
            result = result["_source"]["object"]
            reference = result["reference"]
            prop_id = result["id"]
            url = f"https://iam.dewaele.com/#/portfolio/properties/{prop_id}/general"
            response += f"* {reference}\n  URL: {url}\n"

    return response


functions = {"iam_calendar": get_calendar, "iam_properties": find_properties}


class DewaeleLLM(AzureOpenAILLM):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.functions = [
            cast(
                ChatCompletionToolParam,
                Function(
                    name="iam_calendar",
                    description="Haal informatie en afspraken uit de agenda van de huidige gebruiker.",
                    parameters=FunctionParameters(
                        required_properties=[],
                        optional_properties=[],
                    ),
                ).dict(),
            ),
            cast(
                ChatCompletionToolParam,
                Function(
                    name="iam_properties",
                    description="Zoek naar panden in IAM op basis van sleutelwoorden.",
                    parameters=FunctionParameters(
                        required_properties=[],
                        optional_properties=[
                            FunctionPrimitiveProperty(
                                name="keywords",
                                type="string",
                                description="Keywords to search for",
                            )
                        ],
                    ),
                ).dict(),
            ),
        ]

    def chat_completion(
        self,
        messages: Messages,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        model_params_dict: Dict[str, Any] = deepcopy(self.default_model_params)
        model_params_dict.update(model_params or {})

        message_dict = [m.dict() for m in messages]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=message_dict,
            stream=stream,
            max_tokens=max_tokens,
            functions=self.functions,
            function_call="auto",
            **model_params_dict,
        )

        def streaming_iterator(messages, response):
            func_call = {
                "name": None,
                "arguments": "",
            }
            for chunk in response:
                data = StreamingChatChunk.parse_obj(chunk)
                if data.choices and data.choices[0].delta["function_call"] is None:
                    if data.choices[0].finish_reason == "function_call":
                        # function call here using func_call
                        print("Calling function", func_call)
                        func = functions[func_call["name"]]
                        arguments = json.loads(func_call["arguments"])
                        function_result = func(**arguments)
                        messages.append(
                            AssistantMessage(
                                function_call={
                                    "name": func_call["name"],
                                    "arguments": func_call["arguments"],
                                },
                                content="",
                            )
                        )
                        messages.append(
                            FunctionMessage(
                                name=func_call["name"],
                                content=function_result,
                            )
                        )
                        for response in self.chat_completion(
                            messages=messages,
                            stream=stream,
                            max_tokens=max_tokens,
                            model_params=model_params,
                        ):
                            yield response
                    yield data
                    continue

                if not data.choices:
                    yield data
                    continue

                delta = data.choices[0].delta
                if "function_call" in delta:
                    if delta["function_call"].name:
                        func_call["name"] = delta["function_call"].name
                    if delta["function_call"].arguments:
                        func_call["arguments"] += delta["function_call"].arguments
                    if data.choices[0].finish_reason:
                        # function call here using func_call
                        print("Calling function", func_call)
                        yield data
                if not delta.get("content", None):
                    continue

                yield data

        if stream:
            return streaming_iterator(messages, response)

        return ChatResponse.parse_obj(response)
