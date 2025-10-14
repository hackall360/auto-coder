---
title: Structured Response
description: Enforce a structured response from the model using Pydantic models or JSON Schema
index: 4
---

You can enforce a particular response format from an LLM by providing a JSON schema to the `.respond()` method.
This guarantees that the model's output conforms to the schema you provide.

The JSON schema can either be provided directly,
or by providing an object that implements the `lmstudio.ModelSchema` protocol,
such as `pydantic.BaseModel` or `lmstudio.BaseModel`.

The `lmstudio.ModelSchema` protocol is defined as follows:

```python
@runtime_checkable
class ModelSchema(Protocol):
    """Protocol for classes that provide a JSON schema for their model."""

    @classmethod
    def model_json_schema(cls) -> DictSchema:
        """Return a JSON schema dict describing this model."""
        ...

```

When a schema is provided, the prediction result's `parsed` field will contain a string-keyed dictionary that conforms
to the given schema (for unstructured results, this field is a string field containing the same value as `content`).


## Enforce Using a Class Based Schema Definition

If you wish the model to generate JSON that satisfies a given schema,
it is recommended to provide a class based schema definition using a library
such as [`pydantic`](https://docs.pydantic.dev/) or [`msgspec`](https://jcristharif.com/msgspec/).

Pydantic models natively implement the `lmstudio.ModelSchema` protocol,
while `lmstudio.BaseModel` is a `msgspec.Struct` subclass that implements `.model_json_schema()` appropriately.

#### Define a Class Based Schema

```lms_code_snippet
  variants:
    "pydantic.BaseModel":
      language: python
      code: |
        from pydantic import BaseModel

        # A class based schema for a book
        class BookSchema(BaseModel):
            title: str
            author: str
            year: int

    "lmstudio.BaseModel":
      language: python
      code: |
        from lmstudio import BaseModel

        # A class based schema for a book
        class BookSchema(BaseModel):
            title: str
            author: str
            year: int

```

#### Generate a Structured Response

```lms_code_snippet
  variants:
    "Non-streaming":
      language: python
      code: |
        result = model.respond("Tell me about The Hobbit", response_format=BookSchema)
        book = result.parsed

        print(book)
        #           ^
        # Note that `book` is correctly typed as { title: string, author: string, year: number }

    Streaming:
      language: python
      code: |
        prediction_stream = model.respond_stream("Tell me about The Hobbit", response_format=BookSchema)

        # Optionally stream the response
        # for fragment in prediction:
        #   print(fragment.content, end="", flush=True)
        # print()
        # Note that even for structured responses, the *fragment* contents are still only text

        # Get the final structured result
        result = prediction_stream.result()
        book = result.parsed

        print(book)
        #           ^
        # Note that `book` is correctly typed as { title: string, author: string, year: number }
```

## Enforce Using a JSON Schema

You can also enforce a structured response using a JSON schema.

#### Define a JSON Schema

```python
# A JSON schema for a book
schema = {
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "author": { "type": "string" },
    "year": { "type": "integer" },
  },
  "required": ["title", "author", "year"],
}
```

#### Generate a Structured Response

```lms_code_snippet
  variants:
    "Non-streaming":
      language: python
      code: |
        result = model.respond("Tell me about The Hobbit", response_format=schema)
        book = result.parsed

        print(book)
        #     ^
        # Note that `book` is correctly typed as { title: string, author: string, year: number }

    Streaming:
      language: python
      code: |
        prediction_stream = model.respond_stream("Tell me about The Hobbit", response_format=schema)

        # Stream the response
        for fragment in prediction_stream:
            print(fragment.content, end="", flush=True)
        print()
        # Note that even for structured responses, the *fragment* contents are still only text

        # Get the final structured result
        result = prediction_stream.result()
        book = result.parsed

        print(book)
        #     ^
        # Note that `book` is correctly typed as { title: string, author: string, year: number }
```

## Overview

Once you have [downloaded and loaded](/docs/LMStudio/app/basics/index) a large language model,
you can use it to respond to input through the API. This article covers getting JSON structured output, but you can also
[request text completions](/docs/LMStudio/developer/typescript/llm-prediction/completion),
[request chat responses](/docs/LMStudio/developer/typescript/llm-prediction/chat-completion), and
[use a vision-language model to chat about images](/docs/LMStudio/developer/typescript/llm-prediction/image-input).

### Usage

Certain models are trained to output valid JSON data that conforms to
a user-provided schema, which can be used programmatically in applications
that need structured data. This structured data format is supported by both
[`complete`](/docs/LMStudio/developer/typescript/llm-prediction/completion) and [`respond`](/docs/LMStudio/developer/typescript/llm-prediction/chat-completion)
methods. In Python you typically rely on Pydantic models or explicit JSON schema
dictionaries to define the target structure.

```lms_code_snippet
  variants:
    "Python":
      language: python
      code: |
        import lmstudio as lms
        from pydantic import BaseModel

        class Book(BaseModel):
            title: str
            author: str
            year: int

        model = lms.llm("llama-3.2-1b-instruct")

        response = model.respond(
            "Tell me about The Hobbit.",
            response_format=Book,
        )

        # response.parsed is a dict that matches the Book schema
        print(response.parsed["title"])
```

### Structured generation caveats

Structured outputs depend on the model's ability to follow schema guidance and on
local validation steps. Keep the following considerations in mind:

* **Schema and prompt alignment.** Large schema objects or optional fields can make
  generation unstable. Provide concise, well-documented schemas and include
  clarifying instructions in the prompt for fields that are frequently missing.
* **Model compatibility.** Only instruction-tuned models that have been optimized
  for JSON-mode generation should be expected to reliably follow schemas. If you
  switch to a different checkpoint, re-run your validation suite to ensure it
  still produces structured JSON.
* **Validation failures.** Both class-based schemas and raw JSON schema responses
  are validated before they are returned. Handle exceptions such as
  `pydantic.ValidationError` or `jsonschema.ValidationError` so that your
  application can retry or fall back gracefully when the model emits malformed
  payloads.

```lms_code_snippet
  variants:
    "Validation handling":
      language: python
      code: |
        import logging
        import lmstudio as lms
        from pydantic import BaseModel, ValidationError

        log = logging.getLogger(__name__)

        class Book(BaseModel):
            title: str
            author: str
            year: int

        model = lms.llm("llama-3.2-1b-instruct")

        result = model.respond(
            "Return metadata about The Hobbit as JSON.",
            response_format=Book,
        )

        try:
            book = Book.model_validate(result.parsed)
        except ValidationError as exc:
            # Retry with a stricter prompt or fall back to unstructured output
            log.warning("Structured response failed validation: %s", exc)
            book = None
```
