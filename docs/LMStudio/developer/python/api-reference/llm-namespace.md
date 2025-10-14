---
title: "`client.llm`"
sidebar_title: "`client.llm` namespace"
description: "`client.llm` - API reference for the llm namespace in an `LMStudioClient` instance"
index: 6
---

The `llm` namespace exposes operations for discovering, loading, and interacting with large language
models hosted by LM Studio. It is available on both the synchronous [`Client`](./lmstudioclient.md)
and asynchronous `AsyncClient` classes as the `client.llm` property. The same helpers are also
re-exported as top-level convenience functions such as `lmstudio.llm()`.

```python
import lmstudio as lms

with lms.Client() as client:
    llama = client.llm.model("lmstudio-community/llama-3.2-3b-instruct")
    print(llama.respond("Hello there!"))
```

All interactions ultimately yield an [`LLM` handle](./model.md) whose methods perform predictions
(`.respond()`, `.complete()`, `.act()`, `.tokenize()`, ...). This page focuses on the namespace-level
helpers that orchestrate model lifecycle.

## Core operations

| Method | Description |
| --- | --- |
| [`model(model_key=None, *, ttl=..., config=None, on_load_progress=None)`](#retrieve-a-handle) | Return a handle to an already loaded model or trigger a just-in-time load. |
| [`load_new_instance(model_key, instance_identifier=None, *, ttl=..., config=None, on_load_progress=None)`](#load-additional-model-instances) | Force a fresh copy of the model into memory even when one is already resident. |
| [`list_loaded()`](#inspect-loaded-models) | Enumerate every LLM currently in memory. |
| [`list_downloaded()`](#discover-downloaded-models) | Return downloaded models that can be loaded without contacting the repository. |
| [`get_model_info(model_specifier)`](#inspect-loaded-models) | Fetch metadata (identifier, context length, etc.) about an in-memory model. |
| [`unload(model_identifier)`](#unload-a-model) | Proactively free GPU/CPU memory for a loaded instance. |
| [`connect()` / `disconnect()`](#manual-session-control) | Manually manage the underlying websocket session (rarely required). |

Every method above is available in synchronous form (`client.llm`) and as `await`-able counterparts on
`AsyncClient.llm`. Keyword parameters are the same across both APIs.

### Retrieve a handle

```python
model = client.llm.model()
```

- Without arguments the first loaded model is returned; an error is raised if none are loaded.
- Provide `model_key` to specify which model should be returned or loaded. The key matches the
  identifiers shown in LM Studio (for example `"lmstudio-community/phi-3.5-mini-instruct"`).
- `config` and `ttl` mirror the parameters described in
  [load-time configuration](../../llm-prediction/parameters.md#load-parameters). They are applied only
  if the model needs to be loaded during the call. See [`model()`](./model.md) for full details.
- `on_load_progress` receives streaming progress events while the server loads the model. The callback
  signature is documented in [Download progress callbacks](../model-management/download-models.md#track-download-progress).

To obtain a handle outside of a scoped `Client`, call the top-level convenience wrapper:

```python
llm = lms.llm("llama-3.2-1b-instruct")
print(llm.complete("One-sentence summary of LM Studio"))
```

### Load additional model instances

```python
fresh = client.llm.load_new_instance(
    "lmstudio-community/llama-3.2-3b-instruct",
    instance_identifier="analysis",
    ttl=15 * 60,
    config={"contextLength": 8192},
)
```

`load_new_instance()` always creates a new runtime instance, allowing you to run multiple copies of
the same base model simultaneously (for example, one dedicated to low-latency streaming responses and
another to long-form analysis). Use [`list_loaded()`](#inspect-loaded-models) to view the identifiers
you can pass to [`unload()`](#unload-a-model) later.

### Inspect loaded models

```python
for handle in client.llm.list_loaded():
    info = client.llm.get_model_info(handle.identifier)
    print(info.display_name, info.context_length)
```

`list_loaded()` returns `LLM` handles representing each in-memory model. `get_model_info()` accepts a
model identifier string or any other [model specifier](../../model-info/get-model-info.md#parameters)
and yields a `ModelInstanceInfo` struct.

### Discover downloaded models

```python
for downloaded in client.llm.list_downloaded():
    print(downloaded.display_name, downloaded.path)
```

The return value is a sequence of `DownloadedLlm` wrappers. Each wrapper exposes:

- `model_key`, `display_name`, `architecture`, and other metadata through the `info` property
- `.model(...)` – a shortcut to `client.llm.model(model_key, ...)`
- `.load_new_instance(...)` – identical to calling the namespace method directly

To search the online model repository instead, use [`client.repository`](../model-management/download-models.md).

### Unload a model

```python
client.llm.unload("analysis")
```

Pass the instance identifier returned by [`list_loaded()`](#inspect-loaded-models). Unloading frees
resources immediately instead of waiting for the [idle TTL](../../app/api/ttl-and-auto-evict.md) to
expire.

### Preview a prompt template

```python
rendered = llama.apply_prompt_template(chat, opts={"preset": "instruct"})
print(rendered)
```

The handle method `LLM.apply_prompt_template()` accepts any chat history supported by
[`LLM.respond()`](./respond.md) and returns the formatted prompt string without performing a
prediction. This is handy for debugging how a conversation will be rendered before streaming tokens.
For higher-level guidance see
[Apply prompt templates](../../more/apply-prompt-template.md).

### Streaming predictions

Use the `*_stream()` helpers on an `LLM` handle to receive partial outputs while a prediction is in
progress.

```python
stream = llama.respond_stream("Explain speculative decoding")
for fragment in stream:
    print(fragment.content, end="")
```

- [`LLM.respond_stream()`](./respond.md#streaming) accepts either a `Chat` instance or anything that
  [`Chat.from_history()`](./chat.md#clone-or-restore-an-existing-history) can understand.
- [`LLM.complete_stream()`](./complete.md#streaming) exposes the same pattern for single-turn prompts.
- In the asynchronous API these helpers return an `AsyncPredictionStream` that you iterate with
  `async for`.

Consult [Chat completions](../../llm-prediction/chat-completion.md#streaming) for a full walkthrough,
including how to merge streamed events back into a [`Chat`](./chat.md).

### Manual session control

The namespace automatically opens and closes its websocket connection as needed. In advanced
integrations you can explicitly control the lifecycle:

```python
session = client.llm.connect()
try:
    ...  # Issue multiple remote_call(...) invocations
finally:
    client.llm.disconnect()
```

Maintaining a persistent connection can reduce latency when issuing a large batch of predictions or
invoking lower-level RPCs through `client.llm.remote_call(endpoint, params)`.

## Asynchronous usage

The asynchronous API mirrors the synchronous surface area:

```python
import lmstudio as lms

async with lms.AsyncClient() as client:
    model = await client.llm.model("qwen2.5-7b-instruct")
    stream = await model.respond_stream("Hi there")
    async for fragment in stream:
        print(fragment.content, end="")
```

Async methods return awaitables (`await client.llm.list_loaded()`, `await client.llm.unload(...)`,
etc.) and the streaming helpers yield `AsyncPredictionStream` instances. The semantics of `config`,
`ttl`, and callbacks remain identical to their synchronous counterparts.

## See also

- [`client.system`](./system-namespace.md) for accessing the shared catalogue of downloaded models.
- [Load models into memory](../model-management/loading.md) for higher-level lifecycle patterns.
- [Chat completion guide](../../llm-prediction/chat-completion.md) and
  [text completion guide](../../llm-prediction/completion.md) for prediction workflows once you have
  an `LLM` handle.
