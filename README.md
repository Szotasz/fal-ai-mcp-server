# fal.ai MCP Server

An MCP (Model Context Protocol) server that gives Claude access to [fal.ai](https://fal.ai) generative AI models — images, videos, audio, 3D models, and more.

Built with [FastMCP](https://github.com/jlowin/fastmcp). Works with Claude Code and Claude Desktop.

## Model Catalog

### Image Generation (10 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `flux-dev` | FLUX.1 Dev | High-quality balanced generation | ~$0.025/image |
| `flux-schnell` | FLUX.1 Schnell | Fast generation (4 steps) | ~$0.003/image |
| `flux-pro` | FLUX.1 Pro v1.1 | Best quality FLUX model | ~$0.05/image |
| `flux-kontext` | FLUX Kontext Max | Context-aware editing, multi-image input | ~$0.08/image |
| `recraft-v4` | Recraft V4 | Typography and design-oriented | ~$0.04/image |
| `nano-banana` | Nano Banana | Ultra-fast, ultra-cheap | ~$0.001/image |
| `nano-banana-2` | Nano Banana 2 | Gemini 3.1 Flash — text rendering, consistency | ~$0.08/image |
| `nano-banana-pro` | Nano Banana Pro | Gemini 3 Pro — state-of-the-art | ~$0.15/image |
| `gpt-image` | GPT-Image 1.5 | OpenAI — high-fidelity, strong prompt adherence | ~$0.13/image |
| `seedream-v4` | Seedream V4 | ByteDance — up to 4K resolution | ~$0.03/image |

### Image Editing (4 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `nano-banana-2-edit` | Nano Banana 2 Edit | Gemini 3.1 Flash — fast editing, text rendering | ~$0.08/edit |
| `nano-banana-pro-edit` | Nano Banana Pro Edit | Gemini 3 Pro — semantic editing, 14 ref images | ~$0.15/edit |
| `gpt-image-edit` | GPT-Image 1.5 Edit | OpenAI — preserves composition and lighting | ~$0.13/edit |
| `seedream-v4-edit` | Seedream V4 Edit | ByteDance — context-aware editing | ~$0.03/edit |

### Video Generation (5 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `kling-v3-pro` | Kling V3 Pro | High-quality text/image to video | ~$0.30/video |
| `sora-2-t2v` | Sora 2 Text-to-Video | OpenAI Sora text-to-video | ~$0.50/video |
| `sora-2-i2v` | Sora 2 Image-to-Video | OpenAI Sora image-to-video | ~$0.50/video |
| `ltx-2` | LTX Video V2 | Fast, affordable video | ~$0.05/video |
| `wan-v2` | Wan 2.1 | Alibaba's video model | ~$0.10/video |

### Audio Generation (4 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `chatterbox-tts` | Chatterbox TTS | High-quality text-to-speech | ~$0.01/gen |
| `minimax-speech` | MiniMax Speech 02 HD | HD multi-voice synthesis | ~$0.02/gen |
| `beatoven-music` | Beatoven Music | AI music from text | ~$0.05/gen |
| `beatoven-sfx` | Beatoven SFX | Sound effects | ~$0.03/gen |

### 3D Generation (2 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `meshy-v6` | Meshy V6 | Image to 3D model | ~$0.15/model |
| `trellis-2` | Trellis 2 | Fast image to 3D with textures | ~$0.10/model |

### Utility (2 models)

| Short Name | Model | Description | Price |
|---|---|---|---|
| `remove-bg` | Background Removal | Remove image background | ~$0.005/image |
| `upscale` | Aura SR Upscale | 4x image upscaling | ~$0.01/image |

## Available Tools

| Tool | Description |
|---|---|
| `generate_image` | Generate images from text prompts |
| `edit_image` | Edit/transform existing images |
| `generate_video` | Generate video from text and/or image |
| `generate_audio` | Text-to-speech, music, and sound effects |
| `generate_3d` | Generate 3D models from images |
| `remove_background` | Remove image backgrounds |
| `upscale_image` | Upscale images to higher resolution |
| `run_model` | Run any fal.ai model with custom arguments |
| `list_models` | List all available models from the catalog |

## Installation

### Prerequisites

- Python 3.10+
- A [fal.ai](https://fal.ai) account with an API key

### 1. Get your fal.ai API key

Go to [fal.ai Dashboard > Keys](https://fal.ai/dashboard/keys) and create an API key.

### 2. Set up the server

```bash
git clone https://github.com/Szotasz/fal-ai-mcp-server.git
cd fal-ai-mcp-server
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export FAL_KEY="your-api-key-here"
```

## Usage with Claude Code

```bash
claude mcp add fal-ai \
  -e FAL_KEY=your-api-key-here \
  -- /path/to/fal-ai-mcp-server/.venv/bin/python /path/to/fal-ai-mcp-server/server.py
```

## Usage with Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fal-ai": {
      "command": "/path/to/fal-ai-mcp-server/.venv/bin/python",
      "args": ["/path/to/fal-ai-mcp-server/server.py"],
      "env": {
        "FAL_KEY": "your-api-key-here"
      }
    }
  }
}
```

Replace `/path/to/fal-ai-mcp-server` with the actual path where you cloned the repo.

## Output

Generated files are automatically downloaded to `~/Downloads/fal-ai/` organized by type:
- `images/` — generated and edited images
- `videos/` — generated videos
- `audio/` — speech, music, and sound effects
- `3d/` — 3D models
- `other/` — everything else

## License

MIT
