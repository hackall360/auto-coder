---
title: "Introduction to Plugins"
description: "A brief introduction to making plugins for LM Studio using TypeScript."
index: 1
---

```lms_private_beta
Plugin support is currently in private beta. [Join the beta here](https://forms.gle/ZPfGLMvVC6DbSRQm9).
```

Plugins extend LM Studio's functionality by providing "hook functions" that execute at specific points during operation.

Plugins are currently written in JavaScript/TypeScript and run on Node.js v20.18.0. Python support is in development.

## Getting Started

LM Studio includes Node.js, so no separate installation is required.

### Create a new plugin

LM Studio stores plugins in your workspace directory (open **Developer → Plugins → Open plugins folder** to jump there). The following commands scaffold a minimal TypeScript plugin in that location:

```bash
mkdir hello-world-plugin
cd hello-world-plugin
npm init -y
npm install @lmstudio/sdk
npm install --save-dev typescript @types/node
```

After installing dependencies, create the typical project layout:

```
hello-world-plugin/
├── manifest.json
├── package.json
├── src/
│   ├── index.ts
│   └── toolsProvider.ts
└── tsconfig.json
```

### Understand the manifest

The `manifest.json` file tells LM Studio how to load your plugin. A minimal manifest looks like this:

```json
{
  "name": "hello-world-plugin",
  "displayName": "Hello World",
  "version": "0.1.0",
  "description": "Logs a welcome message when a chat opens.",
  "main": "dist/index.js",
  "owner": "your-hub-handle"
}
```

- `name` must be kebab-case and uniquely identifies the plugin.
- `displayName` and `description` appear in LM Studio.
- `main` points to the compiled JavaScript file that exports your plugin's entry point.
- `owner` determines whose LM Studio Hub namespace the plugin belongs to when you publish it.

### Implement your first hook

Add a TypeScript entry point in `src/index.ts` and a simple tools provider implementation in `src/toolsProvider.ts`:

```lms_code_snippet
  title: "src/toolsProvider.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { tool, type Tool, type ToolsProviderController } from "@lmstudio/sdk";

        export async function toolsProvider(ctl: ToolsProviderController): Promise<Tool[]> {
          const helloTool = tool({
            name: "say_hello",
            description: "Reply with a friendly greeting.",
            implementation: async () => "👋 Hello from the Hello World plugin!"
          });

          ctl.logger.info("Registered the say_hello tool");
          return [helloTool];
        }
```

```lms_code_snippet
  title: "src/index.ts"
  variants:
    TypeScript:
      language: typescript
      code: |
        import { PluginContext } from "@lmstudio/sdk";
        import { toolsProvider } from "./toolsProvider";

        export async function main(context: PluginContext) {
          context.withToolsProvider(toolsProvider);
        }
```

The exported `main` function is called by LM Studio when your plugin loads. Inside it you register plugin features—`withToolsProvider` wires the `say_hello` tool into the chat experience. Explore other hooks in the dedicated feature guides listed below.

### Configure TypeScript and build scripts

Generate a `tsconfig.json` tuned for the plugin build output:

```bash
npx tsc --init --rootDir src --outDir dist --module nodenext --moduleResolution nodenext --esModuleInterop --resolveJsonModule
```

Then add a build script to `package.json`:

```json
{
  "scripts": {
    "build": "tsc --project tsconfig.json"
  }
}
```

Compile the plugin with:

```bash
npm run build
```

The command emits `dist/index.js`, the file referenced by `main` in `manifest.json`.

### Debugging tips

- Use `console.log` or the structured `ctl.logger` helpers from the controllers passed to your hooks; their output is streamed to the terminal when you run `lms dev`.
- Re-run `npm run build` whenever you change TypeScript files outside of development mode. You can watch for changes automatically with `npx tsc --watch`.
- To attach a debugger, prefix commands with `NODE_OPTIONS='--inspect'` to enable the Node.js inspector: `NODE_OPTIONS='--inspect' lms dev`.

### Run a plugin in development mode

Once you've created a plugin, run this command in the plugin directory to start development mode:

```bash
lms dev
```

Your plugin will appear in LM Studio's plugin list. Development mode automatically rebuilds and reloads your plugin when you make code changes.

You only need `lms dev` during development. When the plugin is installed, LM Studio automatically runs them as needed. Learn more about distributing and installing plugins in the [Sharing Plugins](./publish-plugins) section.

## Next Steps

- [Tool Provider](./tool-provider)

  Give models extra capabilities by creating tools they can use during generation, like accessing external APIs or performing calculations.

- [Prompt Preprocessors](./prompt-preprocessor)

  Modify user input before it reaches the model - handle file uploads, inject context, or transform queries.

- [Generators](./generator)

  Create custom text generation sources that replace the local model, perfect for online model adapters.

- [Custom Configurations](./custom-configuration)

  Add configuration UIs so users can customize your plugin's behavior.

- [Third-Party Dependencies](./dependencies)

  Use npm packages to leverage existing libraries in your plugins.

- [Sharing Plugins](./publish-plugins)

  Package and share your plugins with the community.
