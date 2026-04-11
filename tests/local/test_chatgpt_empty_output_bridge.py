from types import SimpleNamespace

from litellm import ModelResponse
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse


def test_empty_output_returns_empty_assistant_response_instead_of_raising():
    handler = LiteLLMResponsesTransformationHandler()
    raw = ResponsesAPIResponse.model_construct(
        output=[],
        id="resp_123",
        created_at=0,
        model="gpt-5.4",
    )
    model_response = ModelResponse()
    logging_obj = SimpleNamespace(post_call=lambda **kwargs: None)

    result = handler.transform_response(
        model="chatgpt/gpt-5.4",
        raw_response=raw,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        encoding=None,
        json_mode=False,
    )

    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].message.content == ""


def test_non_empty_output_does_not_take_empty_output_fallback():
    handler = LiteLLMResponsesTransformationHandler()
    raw = ResponsesAPIResponse.model_construct(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        id="resp_456",
        created_at=0,
        model="gpt-5.4",
    )
    model_response = ModelResponse()
    logging_obj = SimpleNamespace(post_call=lambda **kwargs: None)

    result = handler.transform_response(
        model="chatgpt/gpt-5.4",
        raw_response=raw,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        encoding=None,
        json_mode=False,
    )

    assert result.choices[0].message.content == "hello"
