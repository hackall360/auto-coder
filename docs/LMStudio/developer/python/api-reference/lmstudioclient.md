---
title: "`LMStudioClient`"
sidebar_title: "`LMStudioClient`"
description: "LMStudioClient - API reference for the `LMStudioClient` class"
index: 1
---

`LMStudioClient` is the root object of the LM Studio Python SDK. It encapsulates the websocket
connection to the desktop runtime, manages authentication, and surfaces the typed namespaces used to
load models, run predictions, inspect the host, and coordinate downloads. Most applications create a
single client per process and reuse it for the duration of their workload.

## Creating a client

The default constructor discovers a locally running LM Studio desktop instance and negotiates a
session automatically. The recommended pattern is to rely on the synchronous context manager so the
connection is closed even if an exception bubbles out of the scope:

```python
import lmstudio as lms

with lms.Client() as client:
    llama = client.llm.model("lmstudio-community/llama-3.2-3b-instruct")
    summary = llama.complete("One sentence about LM Studio.")
    print(summary.content)
    llama.unload()
```

Behind the scenes `Client.__enter__()` opens the websocket and `Client.__exit__()` calls
`client.close()`. If you prefer manual control, instantiate `lms.Client()` and invoke `close()`
explicitly inside a `try`/`finally` block.

### Asynchronous clients

For asyncio-based services the SDK offers `AsyncClient`, which mirrors the synchronous API and is
compatible with `async with`:

```python
import asyncio
import lmstudio as lms

async def main():
    async with lms.AsyncClient() as client:
        model = await client.llm.model("lmstudio-community/qwen2.5-7b-instruct")
        stream = await model.respond_stream("List three productivity tips.")
        async for event in stream:
            if event.delta:
                print(event.delta, end="")
        await model.unload()

asyncio.run(main())
```

The asynchronous client lazily opens the websocket when the first RPC is executed and closes it when
the scope exits. Both `Client` and `AsyncClient` can also be combined with `contextlib.ExitStack`
or `AsyncExitStack` when you need to coordinate multiple managed resources. For example, an
asynchronous workflow can register cleanup hooks to guarantee models are unloaded even if a later
awaitable fails:

```python
from contextlib import AsyncExitStack

async with AsyncExitStack() as stack:
    client = await stack.enter_async_context(lms.AsyncClient())
    llama = await client.llm.model("lmstudio-community/llama-3.2-3b-instruct")
    stack.push_async_callback(llama.unload)
    result = await llama.respond("Explain tool calling in two sentences.")
    print(result.message.content)
```

## Primary namespaces

Each client exposes cohesive namespaces that group related operations. The same namespaces are
available on both synchronous and asynchronous clients:

| Namespace | Description |
| --- | --- |
| [`client.llm`](./llm-namespace.md) | Load, run, and unload language models. Provides chat, completion, and agent workflows. |
| `client.embedding` | Access embedding models for vectorization tasks. |
| `client.files` | Stage binary assets—images, audio, documents—before attaching them to prompts. |
| [`client.system`](./system-namespace.md) | Inspect the host, enumerate downloaded artifacts, and observe runtime health. |
| `client.repository` | Authenticate with LM Studio Hub and query the remote catalogue of downloadable models. |
| `client.diagnostics` | Surface performance counters and low-level telemetry while debugging. |

Top-level shortcuts such as `lmstudio.llm()` and `lmstudio.list_downloaded_models()` forward to the
same RPCs without requiring you to explicitly manage a client object, but the `LMStudioClient`
instance gives you full lifecycle control and shared authentication.

## Authentication and configuration

By default the SDK connects to `ws://127.0.0.1:<port>` and inherits authentication settings from the
LM Studio desktop host. Override the behaviour with constructor arguments or environment variables:

```python
import os

client = lms.Client(
    base_url="wss://studio.internal.example.com:9000",
    client_identifier="batch-service",
    client_passkey=os.environ["LMSTUDIO_CLIENT_PASSKEY"],
)
```

Environment variables provide a convenient way to configure containerised deployments without
hard-coding secrets:

- `LMSTUDIO_BASE_URL` – overrides auto-discovery with a specific websocket endpoint.
- `LMSTUDIO_CLIENT_IDENTIFIER` – labels the session; matching identifiers share server-side caches.
- `LMSTUDIO_CLIENT_PASSKEY` – authenticates the client against a secured LM Studio host.
- `LMSTUDIO_CA_BUNDLE` – path to a custom CA bundle when connecting to TLS-terminated reverse proxies.

When both environment variables and constructor arguments are supplied, explicit arguments win. Pass
`verbose_error_messages=True` during development to surface full server stack traces when the SDK
raises an exception.

## Typical workflow

Most integrations follow the same pattern: acquire a client, ensure the desired model is loaded,
perform predictions, and then free resources when the session ends. The snippet below mirrors the
structure used across the other Python reference pages:

```python
import lmstudio as lms

with lms.Client() as client:
    # 1. Load or retrieve the model you need.
    llama = client.llm.model("lmstudio-community/llama-3.2-3b-instruct", ttl=1800)

    # 2. Interact with the model using the high-level helpers.
    chat = lms.Chat(initial_prompt="You are a concise assistant.")
    chat.add_user_message("Summarise the latest release notes.")
    prediction = llama.respond(chat)
    chat.append(prediction.message)

    # 3. Release resources explicitly (optional when relying on TTL eviction).
    llama.unload()
```

The same workflow adapts naturally to the asynchronous API by replacing `Client` with `AsyncClient`
and awaiting each coroutine. Whenever you exit the context manager (synchronous or asynchronous) the
client closes its websocket and the LM Studio server will reclaim any models whose idle TTL has
expired.

## See also

- [`client.llm`](./llm-namespace.md) – deeper coverage of loading, listing, and unloading language models.
- [`client.system`](./system-namespace.md) – inspect downloaded artifacts and host status.
- [Prediction guides](../../llm-prediction/chat-completion.md) – end-to-end walkthroughs once you have a client and model handle.
