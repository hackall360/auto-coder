---
title: Download Models
description: Download models to the machine running the LM Studio server
---

## Overview

You can browse and download models using the LM Studio SDK just like you would
in the Discover tab of the app itself. Once a model is downloaded, you can
[load it](/docs/LMStudio/developer/typescript/model-management/loading) for inference.

### Usage

Downloading models consists of three steps:

1. Search for the model you want;
2. Find the download option you want (e.g. quantization); and
3. Download the model!

```lms_code_snippet
  variants:
    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient } from "@lmstudio/sdk";

        const client = new LMStudioClient();

        // 1. Search for the model you want
        // Specify any/all of searchTerm, limit, compatibilityTypes
        const searchResults = await client.repository.searchModels({
          searchTerm: "llama 3.2 1b",    // Search for Llama 3.2 1B
          limit: 5,                      // Get top 5 results
          compatibilityTypes: ["gguf"],  // Only download GGUFs
        });

        // 2. Find download options
        const bestResult = searchResults[0];
        const downloadOptions = await bestResult.getDownloadOptions();

        // Let's download Q4_K_M, a good middle ground quantization
        const desiredModel = downloadOptions.find(option => option.quantization === 'Q4_K_M');

        // 3. Download it!
        const modelKey = await desiredModel.download();

        // This returns a path you can use to load the model
        const loadedModel = await client.llm.model(modelKey);
```

## Advanced Usage

### Progress callbacks

The TypeScript SDK accepts camel-cased callback properties (`onProgress` and
`onStartFinalizing`), whereas the Python SDK uses snake-cased keyword arguments
(`on_progress` and `on_finalize`). This mirrors the conventions of each
language, but the payloads delivered to the callbacks are otherwise equivalent.

Model downloading can take a very long time, depending on your local network speed.
If you want to get updates on the progress of this process, you can provide callbacks to `download`:
one for progress updates and/or one when the download is being finalized
(validating checksums, etc.)

```lms_code_snippet
  variants:
    Python (with scoped resources):
      language: python
      code: |
        import lmstudio as lms

        def print_progress_update(update: lms.DownloadProgressUpdate) -> None:
            percent = 100 * update.downloaded_bytes / update.total_bytes
            print(
                f"Downloaded {percent:.1f}% "
                f"({update.downloaded_bytes:,}/{update.total_bytes:,} bytes) "
                f"at {update.speed_bytes_per_second:,.0f} B/s",
                end="\r",
            )

        with lms.Client() as client:
            search_results = client.repository.search_models(
                search_term="llama 3.2 1b",
                limit=5,
                compatibility_types=["gguf"],
            )
            if not search_results:
                raise RuntimeError("No models matched the search term")

            download_options = search_results[0].get_download_options()
            desired_option = next(
                (option for option in download_options if option.info.quantization == "Q4_K_M"),
                None,
            )
            if desired_option is None:
                raise RuntimeError("Desired quantization not available")

            model_key = desired_option.download(
                on_progress=print_progress_update,
                on_finalize=lambda: print("\nFinalizing download..."),
            )

    TypeScript:
      language: typescript
      code: |
        import { LMStudioClient, type DownloadProgressUpdate } from "@lmstudio/sdk";

        function printProgressUpdate(update: DownloadProgressUpdate) {
          process.stdout.write(
            `Downloaded ${update.downloadedBytes} bytes of ${update.totalBytes} total ` +
              `at ${update.speedBytesPerSecond} bytes/sec\r`,
          );
        }

        async function main() {
          const client = new LMStudioClient();

          const searchResults = await client.repository.searchModels({
            searchTerm: "llama 3.2 1b",
            limit: 5,
            compatibilityTypes: ["gguf"],
          });
          if (searchResults.length === 0) {
            throw new Error("No models matched the search term");
          }

          const downloadOptions = await searchResults[0].getDownloadOptions();
          const desiredModel = downloadOptions.find(
            option => option.quantization === "Q4_K_M",
          );
          if (!desiredModel) {
            throw new Error("Desired quantization not available");
          }

          const modelKey = await desiredModel.download({
            onProgress: printProgressUpdate,
            onStartFinalizing: () => console.log("Finalizing..."),
          });

          await client.llm.model(modelKey);
        }

        void main();
```
