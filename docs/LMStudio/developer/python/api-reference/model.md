---
title: "`.model()`"
sidebar_title: "`.model()`"
description: ".model() - API reference for obtaining a model handle from an `LMStudioClient` instance"
index: 2
---

The `.model()` method returns a handle to a language model that is managed by LM Studio. You can
retrieve whichever model is currently active, or request a specific model to be loaded on demand.
The same API is available on both the synchronous `LMStudioClient` (`client.llm.model`) and the
asynchronous `AsyncLMStudioClient` (`client.llm.model` when `client` was created with
`lmstudio.AsyncClient`).

## Signatures

```python
# Synchronous client
model = client.llm.model(
    key: str | None = None,
    *,
    config: dict | None = None,
    ttl: int | None = None,
)

# Asynchronous client
model = await client.llm.model(
    key: str | None = None,
    *,
    config: dict | None = None,
    ttl: int | None = None,
)
```

`lmstudio.llm(...)` is a convenience alias for `lmstudio.get_default_client().llm.model(...)`. The
async convenience layer exposes the same call through `lmstudio.AsyncClient()`.

## Parameters

| Name | Type | Description |
| --- | --- | --- |
| `key` | `str \| None` | The model identifier to load. When omitted, the method returns the first loaded LLM instance. |
| `config` | `dict \| None` | Optional [load-time configuration](../../llm-prediction/parameters#load-parameters). Only applied if the model is loaded as part of this call. |
| `ttl` | `int \| None` | Idle time (seconds) before an auto-unload occurs. Only takes effect when this call performs a JIT load. See [Idle TTL and Auto-Evict](/docs/LMStudio/app/api/ttl-and-auto-evict). |

## Returns

- **Synchronous client:** an `LLM` handle that exposes prediction helpers such as
  `.respond()`, `.complete()`, `.tokenize()`, etc.
- **Asynchronous client:** an `AsyncLLM` handle that provides the same methods but returning
  awaitables.

## Usage patterns

### Access whichever model is already loaded

```python
import lmstudio as lms

model = lms.llm()  # Uses the first loaded model, raises an error if none are available.
```

This call never triggers a load; it only returns a handle to an in-memory model. If no model has been
loaded through LM Studio yet, an error is raised prompting you to load one first.

### Request a specific model by key

```python
with lms.Client() as client:
    model = client.llm.model("llama-3.2-1b-instruct")
```

- If the requested model is already resident, the existing instance is returned immediately.
- If it is not yet loaded, LM Studio performs a JIT load using the optional `config` and `ttl`
  arguments.

### Use the asynchronous API

```python
import lmstudio as lms

async with lms.AsyncClient() as client:
    model = await client.llm.model(
        "qwen2.5-7b-instruct",
        config={"contextLength": 8192},
    )
    reply = await model.respond("Say hi!", stream=True)
```

Async calls are especially helpful when coordinating multiple model loads or predictions from an
asyncio application.

## Caching, JIT loading, and TTL behaviour

- **JIT loading** – Calling `.model(key)` will load the model if necessary. Load-time options supplied
  via `config` or `ttl` are honored only during this first load. Subsequent `.model(key)` calls reuse
  the cached instance and ignore new load-time parameters.
- **Instance caching** – LM Studio keeps one active instance per identifier. To spawn an additional
  copy, use [`load_new_instance()`](../model-management/loading#load-a-new-instance-of-a-model-with-load_new_instance).
- **Idle eviction** – Models loaded on demand inherit the workspace’s default idle TTL (60 minutes by
  default, or whatever you configured). Override it per call with the `ttl` argument. Learn more in
  [Idle TTL and Auto-Evict](/docs/LMStudio/app/api/ttl-and-auto-evict).

## See also

- [Manage Models in Memory](../model-management/loading) for higher-level loading and unloading
  patterns.
- [Configure load-time parameters](../../llm-prediction/parameters#load-parameters) for the full list
  of supported `config` fields.
- [Idle TTL and Auto-Evict](/docs/LMStudio/app/api/ttl-and-auto-evict) for more about automatic
  unloading.
