---
title: Download Models
description: Download models to the machine running the LM Studio server
---

## Overview

You can browse and download models using the LM Studio Python SDK in much the same
way that you browse the Discover tab inside the desktop application. The
repository API lets you search the online catalogue, inspect the variants that are
available for download (for example different quantizations of the same base
model), and then download the variant that best suits your hardware.

Once a model is downloaded you can [load it for inference](./loading) using any of
the APIs described in the rest of the Python documentation set.

### Usage

Downloading a model typically involves three steps:

1. Search for the model you want.
2. Choose the download option that best suits your machine.
3. Download the weights and then load them like any other local model.

The snippets below show the same flow using each of the Python SDK styles.

```lms_code_snippet
  variants:
    "Python (convenience API)":
      language: python
      code: |
        import lmstudio as lms

        client = lms.get_default_client()

        # 1. Search for the model you want
        search_results = client.repository.search_models(
            search_term="llama 3.2 1b",   # Search for Llama 3.2 1B
            limit=5,                      # Look at the first 5 matches
            compatibility_types=["gguf"], # Only consider GGUF downloads
        )

        # 2. Pick a download option from the best match
        best_result = search_results[0]
        download_options = best_result.get_download_options()

        desired_option = next(
            option for option in download_options
            if option.info.quantization == "Q4_K_M"
        )

        # 3. Download it, then load it like any other local model
        model_key = desired_option.download()
        model = lms.llm(model_key)

    "Python (scoped resource API)":
      language: python
      code: |
        import lmstudio as lms

        with lms.Client() as client:
            search_results = client.repository.search_models(
                search_term="llama 3.2 1b",
                limit=5,
                compatibility_types=["gguf"],
            )

            best_result = search_results[0]
            download_options = best_result.get_download_options()

            desired_option = next(
                option for option in download_options
                if option.info.quantization == "Q4_K_M"
            )

            model_key = desired_option.download()
            model = client.llm.model(model_key)

    "Python (asynchronous API)":
      language: python
      code: |
        # Note: assumes use of an async function or the "python -m asyncio" asynchronous REPL
        # Requires Python SDK version 1.5.0 or later
        import lmstudio as lms

        async with lms.AsyncClient() as client:
            search_results = await client.repository.search_models(
                search_term="llama 3.2 1b",
                limit=5,
                compatibility_types=["gguf"],
            )

            best_result = search_results[0]
            download_options = await best_result.get_download_options()

            desired_option = next(
                option for option in download_options
                if option.info.quantization == "Q4_K_M"
            )

            model_key = await desired_option.download()
            model = await client.llm.model(model_key)
```

`ModelDownloadOption.info` exposes structured metadata about the download choice,
including fields like `.name`, `.quantization`, `.size_bytes`, and `.fit_estimation`.
You can use those properties to present richer selection UIs or to implement your
own heuristics for picking a preferred quantization.

## Advanced Usage

### Track download progress

Model downloads can take a long time, so the repository APIs accept callbacks that
report download progress and notify you when the server begins the finalization
step (checksum validation and model registration).

- The progress callback signature is `Callable[[DownloadProgressUpdate], None]`.
  The update object exposes `.downloaded_bytes`, `.total_bytes`, and
  `.speed_bytes_per_second`.
- The finalization callback signature is `Callable[[], None]`.

In Python these callbacks are passed to `download()` as the `on_progress` and
`on_finalize` keyword arguments. (In TypeScript, the equivalent names are
`onProgress` and `onStartFinalizing`.)

```lms_code_snippet
  variants:
    "Python (convenience API)":
      language: python
      code: |
        import lmstudio as lms

        def print_progress(update: lms.DownloadProgressUpdate) -> None:
            percent = 100 * update.downloaded_bytes / update.total_bytes
            print(
                f"Downloaded {percent:.1f}% "
                f"({update.downloaded_bytes:,}/{update.total_bytes:,} bytes) "
                f"at {update.speed_bytes_per_second:,.0f} B/s",
                end="\r",
            )

        client = lms.get_default_client()
        search_results = client.repository.search_models(search_term="llama 3.2 1b")
        download_options = search_results[0].get_download_options()
        desired_option = download_options[0]

        model_key = desired_option.download(
            on_progress=print_progress,
            on_finalize=lambda: print("\nFinalizing download..."),
        )

    "Python (scoped resource API)":
      language: python
      code: |
        import lmstudio as lms

        def print_progress(update: lms.DownloadProgressUpdate) -> None:
            percent = 100 * update.downloaded_bytes / update.total_bytes
            print(
                f"Downloaded {percent:.1f}% "
                f"({update.downloaded_bytes:,}/{update.total_bytes:,} bytes) "
                f"at {update.speed_bytes_per_second:,.0f} B/s",
                end="\r",
            )

        with lms.Client() as client:
            search_results = client.repository.search_models(search_term="llama 3.2 1b")
            download_options = search_results[0].get_download_options()
            desired_option = download_options[0]

            model_key = desired_option.download(
                on_progress=print_progress,
                on_finalize=lambda: print("\nFinalizing download..."),
            )

    "Python (asynchronous API)":
      language: python
      code: |
        # Note: assumes use of an async function or the "python -m asyncio" asynchronous REPL
        # Requires Python SDK version 1.5.0 or later
        import lmstudio as lms

        def print_progress(update: lms.DownloadProgressUpdate) -> None:
            percent = 100 * update.downloaded_bytes / update.total_bytes
            print(
                f"Downloaded {percent:.1f}% "
                f"({update.downloaded_bytes:,}/{update.total_bytes:,} bytes) "
                f"at {update.speed_bytes_per_second:,.0f} B/s",
                end="\r",
            )

        async with lms.AsyncClient() as client:
            search_results = await client.repository.search_models(search_term="llama 3.2 1b")
            download_options = await search_results[0].get_download_options()
            desired_option = download_options[0]

            model_key = await desired_option.download(
                on_progress=print_progress,
                on_finalize=lambda: print("\nFinalizing download..."),
            )
```

After the download completes, the returned `model_key` can be used with
`client.llm.model()` or `client.embedding.model()` just like any other locally
available model. If you need to list everything that is already on disk, use
[`list_downloaded_models()`](./list-downloaded) from the same API style that your
application is using.
