from __future__ import annotations

import pytest

from internal.schemas import SchemaError
from internal.structures import PredictionResultAdapter, StructuredResponse


def test_structured_response_parses_content_mapping():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": '{"title": "The Hobbit", "year": 1937}'}
                    ],
                    "parsed": {"title": "The Hobbit", "year": 1937},
                }
            }
        ]
    }

    result = StructuredResponse.from_response(response, structured=True)

    assert result.structured is True
    assert result.content == '{"title": "The Hobbit", "year": 1937}'
    assert result.parsed == {"title": "The Hobbit", "year": 1937}


def test_unstructured_response_leaves_parsed_none():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello world",
                }
            }
        ]
    }

    result = StructuredResponse.from_response(response)

    assert result.structured is False
    assert result.content == "Hello world"
    assert result.parsed is None


def test_structured_response_invalid_payload_raises_schema_error():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "   ",
                }
            }
        ]
    }

    with pytest.raises(SchemaError):
        StructuredResponse.from_response(response, structured=True)


def test_prediction_result_adapter_streaming_accumulates_and_parses():
    adapter = PredictionResultAdapter(schema={"type": "object", "properties": {"value": {"type": "number"}}})

    adapter.add_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "content": [{"type": "output_text", "text": "{"}],
                    }
                }
            ]
        }
    )
    adapter.add_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "content": [{"type": "output_text", "text": '"value": 1}'}],
                    }
                }
            ]
        }
    )

    final_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"value": 1}',
                }
            }
        ]
    }

    result = adapter.result(final_response)

    assert result.content == '{"value": 1}'
    assert result.parsed == {"value": 1}


def test_prediction_result_adapter_without_final_response_uses_chunks():
    adapter = PredictionResultAdapter()

    adapter.add_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {"type": "output_text", "text": "Hello "},
                            {"type": "output_text", "text": "world"},
                        ]
                    }
                }
            ]
        }
    )

    result = adapter.result()

    assert result.parsed is None
    assert result.content == "Hello world"
