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
    assert payload["schema"]["required"] == ["name"]


def test_normalize_schema_rejects_invalid_json():
    try:
        normalize_schema("{}[]")
    except SchemaError:
        return
    assert False, "Expected SchemaError for invalid JSON"


def test_parse_structured_content_from_string():
    data = parse_structured_content('{"name": "Alice"}')
    assert data == {"name": "Alice"}
