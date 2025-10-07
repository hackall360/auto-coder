import json

import pytest

from internal.schemas import (
    SchemaError,
    build_response_format,
    normalize_schema,
    parse_structured_content,
)


class ExampleSchema:
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }


def test_build_response_format_from_mapping():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    }
    response_format = build_response_format(schema, name="value_schema", strict=False)
    assert response_format["type"] == "json_schema"
    payload = response_format["json_schema"]
    assert payload["name"] == "value_schema"
    assert payload["strict"] is False
    assert payload["schema"]["properties"]["value"]["type"] == "number"


def test_build_response_format_from_model_class():
    response_format = build_response_format(ExampleSchema)
    payload = response_format["json_schema"]
    assert payload["name"] == "ExampleSchema"
    assert payload["strict"] is True
    assert payload["schema"]["required"] == ["name"]


def test_build_response_format_derives_name_from_schema_metadata():
    schema = {
        "type": "object",
        "title": "FriendlySchema",
        "properties": {},
    }
    payload = build_response_format(schema)["json_schema"]
    assert payload["name"] == "FriendlySchema"


def test_build_response_format_from_schema_string():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
    )
    payload = build_response_format(schema, strict=True)["json_schema"]
    assert payload["schema"]["properties"]["flag"]["type"] == "boolean"
    assert payload["strict"] is True


def test_normalize_schema_rejects_invalid_json():
    with pytest.raises(SchemaError, match="valid JSON"):
        normalize_schema("{}[]")


def test_normalize_schema_rejects_empty_string():
    with pytest.raises(SchemaError, match="must not be empty"):
        normalize_schema("   ")


def test_normalize_schema_rejects_non_object_json():
    with pytest.raises(SchemaError, match="JSON object"):
        normalize_schema("[]")


def test_normalize_schema_rejects_missing_type_information():
    with pytest.raises(SchemaError, match="type information"):
        normalize_schema({"properties": {}})


def test_normalize_schema_from_model_instance():
    instance = ExampleSchema()
    normalized = normalize_schema(instance)
    assert normalized["required"] == ["name"]


def test_parse_structured_content_from_string():
    data = parse_structured_content('{"name": "Alice"}')
    assert data == {"name": "Alice"}


def test_parse_structured_content_rejects_non_object():
    with pytest.raises(SchemaError, match="JSON object"):
        parse_structured_content("[]")
