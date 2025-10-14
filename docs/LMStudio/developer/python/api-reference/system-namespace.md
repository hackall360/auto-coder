---
title: "`client.system`"
sidebar_title: "`client.system` namespace"
description: "`client.system` - API reference for the system namespace in an `LMStudioClient` instance"
index: 6
---

The `system` namespace exposes LM Studio server-level utilities that apply across all model types.
When you need a catalogue of every model the host has already downloaded—regardless of whether it is
an LLM or embedding model—start here.

```python
import lmstudio as lms

with lms.Client() as client:
    downloaded = client.system.list_downloaded_models()
    for model in downloaded:
        print(model.model_key, model.type, model.path)
```

Returned objects are wrappers around the JSON payloads sent by LM Studio, providing strongly-typed
helpers to load models directly from the result set.

## `list_downloaded_models()`

```python
models = client.system.list_downloaded_models()
```

- Returns a list of `DownloadedLlm` and `DownloadedEmbeddingModel` instances (alias `AnyDownloadedModel`).
- Each wrapper exposes:
  - `model_key`, `display_name`, `architecture`, `vision`, and related metadata through the `.info`
    property.
  - `.model(...)` – equivalent to calling [`client.llm.model()`](./llm-namespace.md#retrieve-a-handle)
    or [`client.embedding.model()`](../embedding/index.md) depending on the model type.
  - `.load_new_instance(...)` – identical to the respective namespace’s
    [`load_new_instance()`](./llm-namespace.md#load-additional-model-instances) helper.
  - `.path` – absolute filesystem path of the downloaded weights, useful for troubleshooting.
- The ordering matches the server response: newly downloaded models appear last.

Use these helpers to build dashboards or reconcile on-disk models with what is currently loaded into
memory. When combined with [`client.llm.list_loaded()`](./llm-namespace.md#inspect-loaded-models) you
can easily detect which downloads are idle.

## Typical workflow

1. Call `client.system.list_downloaded_models()` to enumerate everything on disk.
2. Filter the returned collection based on metadata, e.g. `model.info.architecture == "llama"`.
3. Load a model directly from the wrapper:

    ```python
    llama = next(m for m in models if m.model_key.endswith("llama-3.2-3b-instruct"))
    handle = llama.model(ttl=3600)
    response = handle.respond("Summarise the latest release notes.")
    ```

4. When finished, clean up with [`client.llm.unload(handle.identifier)`](./llm-namespace.md#unload-a-model)
   or allow the idle TTL to release memory automatically.

## Relationship to other namespaces

- [`client.llm`](./llm-namespace.md) and `client.embedding` expose additional lifecycle controls once
you know which model key you want to operate on.
- [`client.repository`](../model-management/download-models.md) queries the online model hub for
resources that are not yet downloaded. Combine it with `client.system` to offer download + load flows
in tooling.
- The top-level helpers [`lmstudio.list_downloaded_models()`](../model-management/download-models.md)
and [`lmstudio.list_loaded_models()`](../model-management/loading.md#list-loaded-models) forward to the
same underlying RPCs if you prefer not to create a scoped client.
