---
title: "`client.system`"
sidebar_title: "`client.system` namespace"
description: "`client.system` - API reference for the system namespace in an `LMStudioClient` instance"
index: 6
---

`client.system` exposes host-level utilities that apply to every runtime managed by LM Studio. Use it
to list downloaded models, watch for disconnections, surface notifications, or control the embedded
HTTP server.

```ts
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
const downloads = await client.system.listDownloadedModels();
console.table(downloads.map(({ modelKey, type, path }) => ({ modelKey, type, path })));
```

Returned items are plain data objects (`ModelInfo`, `LLMInfo`, or `EmbeddingModelInfo`) that you can
pass back into the [`client.llm`](./llm-namespace.md) or `client.embedding` namespaces when you are
ready to load them.

## `listDownloadedModels()`

```ts
const all = await client.system.listDownloadedModels();
const llmsOnly = await client.system.listDownloadedModels("llm");
const embeddingsOnly = await client.system.listDownloadedModels("embedding");
```

- Without arguments the method returns every downloaded artifact, regardless of type.
- Passing `"llm"` or `"embedding"` narrows the result to a single domain.
- Each object includes the fields inherited from `ModelInfoBase`:
  - `modelKey`, `displayName`, `path`, `sizeBytes`, `format`
  - Optional metadata such as `architecture`, `paramsString`, `quantization`
  - For LLMs you also receive `vision`, `trainedForToolUse`, and `maxContextLength`.

You can feed a result back into `client.llm.model(download.modelKey)` to load it without looking up
keys manually. Combine this with [`client.llm.listLoaded()`](./llm-namespace.md#inspect-loaded-models)
to reconcile disk state with in-memory instances.

## `whenDisconnected()`

```ts
await client.system.whenDisconnected();
console.info("Lost connection to LM Studio");
```

Resolves when the underlying websocket connection closes. This is handy for long-lived services that
need to clean up resources once the desktop app exits.

## `notify()`

```ts
await client.system.notify({
  title: "Model loaded",
  description: "llama-3.2-3b-instruct is ready",
});
```

Sends a toast notification to the LM Studio UI. The payload matches `BackendNotification` (`title`,
optional `description`, and `noAutoDismiss` to keep the toast visible until dismissed manually).

## `getLMStudioVersion()`

```ts
const { version, build } = await client.system.getLMStudioVersion();
console.info(`Connected to LM Studio ${version} (build ${build})`);
```

Use this to gate features that depend on specific server capabilities.

## Experimental helpers

> **Note**
> The following methods are marked experimental in the SDK and may change without notice.

### `unstable_setExperimentFlag(flag, value)` / `unstable_getExperimentFlags()`

Toggle or inspect LM Studio's internal experiment flags. These are primarily useful when testing
pre-release features.

```ts
await client.system.unstable_setExperimentFlag("vision-tooling", true);
const enabled = await client.system.unstable_getExperimentFlags();
```

### `startHttpServer(opts)` / `stopHttpServer()`

Programmatically expose LM Studio's HTTP API from the desktop host.

```ts
await client.system.startHttpServer({
  port: 12345,
  cors: true,
});

// … later …
await client.system.stopHttpServer();
```

`startHttpServer` accepts a `StartHttpServerOpts` object with two fields:

| Field | Type | Description |
| --- | --- | --- |
| `port` | `number` | Port on which the API server should listen. Must be available on the host. |
| `cors` | `boolean` | Whether to enable permissive CORS headers for browser-based clients. |

## See also

- [`client.llm`](./llm-namespace.md) for model lifecycle helpers once you know which key to load.
- [List local models guide](../model-management/list-downloaded.md) for higher-level usage patterns.
- [Manage models in memory](../model-management/loading.md) to load or unload items returned by
  `listDownloadedModels()`.
