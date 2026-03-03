"""fal.ai MCP Server — generative AI models for Claude Code."""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import fal_client
import httpx
from fastmcp import FastMCP

mcp = FastMCP("fal-ai", instructions="Generate images, videos, audio, and 3D models using fal.ai")

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "image": {
        "flux-dev": {
            "id": "fal-ai/flux/dev",
            "name": "FLUX.1 Dev",
            "description": "High-quality balanced image generation",
            "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance_scale": 3.5},
            "price": "~$0.025/image",
        },
        "flux-schnell": {
            "id": "fal-ai/flux/schnell",
            "name": "FLUX.1 Schnell",
            "description": "Fast image generation (4 steps)",
            "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 4},
            "price": "~$0.003/image",
        },
        "flux-pro": {
            "id": "fal-ai/flux-pro/v1.1",
            "name": "FLUX.1 Pro v1.1",
            "description": "Best quality FLUX model",
            "defaults": {"width": 1024, "height": 1024, "guidance_scale": 3.5},
            "price": "~$0.05/image",
        },
        "flux-kontext": {
            "id": "fal-ai/flux-pro/kontext/max/multi",
            "name": "FLUX Kontext Max",
            "description": "Context-aware image editing and generation, supports multiple input images",
            "defaults": {},
            "price": "~$0.08/image",
        },
        "recraft-v4": {
            "id": "fal-ai/recraft/v4",
            "name": "Recraft V4",
            "description": "Excellent typography and design-oriented generation",
            "defaults": {"width": 1024, "height": 1024},
            "price": "~$0.04/image",
        },
        "nano-banana": {
            "id": "fal-ai/nano-banana",
            "name": "Nano Banana",
            "description": "Ultra-fast, ultra-cheap image generation",
            "defaults": {"width": 1024, "height": 1024},
            "price": "~$0.001/image",
        },
        "nano-banana-2": {
            "id": "fal-ai/nano-banana-2",
            "name": "Nano Banana 2",
            "description": "Gemini 3.1 Flash — fast, accurate text rendering, character consistency, multi-resolution",
            "defaults": {"resolution": "1k", "aspect_ratio": "1:1"},
            "price": "~$0.08/image",
        },
        "nano-banana-pro": {
            "id": "fal-ai/nano-banana-pro",
            "name": "Nano Banana Pro",
            "description": "Gemini 3 Pro — state-of-the-art generation, semantic understanding, character consistency",
            "defaults": {"resolution": "1k", "aspect_ratio": "1:1"},
            "price": "~$0.15/image",
        },
        "gpt-image": {
            "id": "fal-ai/gpt-image-1.5",
            "name": "GPT-Image 1.5",
            "description": "OpenAI — high-fidelity images, strong prompt adherence, preserves composition and detail",
            "defaults": {"image_size": "1024x1024", "quality": "high"},
            "price": "~$0.13/image (high quality)",
        },
        "seedream-v4": {
            "id": "fal-ai/bytedance/seedream/v4",
            "name": "Seedream V4",
            "description": "ByteDance — unified generation and editing, high resolution up to 4K",
            "defaults": {},
            "price": "~$0.03/image",
        },
    },
    "image_edit": {
        "nano-banana-pro-edit": {
            "id": "fal-ai/nano-banana-pro/edit",
            "name": "Nano Banana Pro Edit",
            "description": "Gemini 3 Pro — semantic image editing, supports up to 14 reference images",
            "defaults": {"resolution": "1k"},
            "price": "~$0.15/edit",
        },
        "gpt-image-edit": {
            "id": "fal-ai/gpt-image-1.5/edit",
            "name": "GPT-Image 1.5 Edit",
            "description": "OpenAI — image editing with high fidelity, preserves composition and lighting",
            "defaults": {"quality": "high", "input_fidelity": "high"},
            "price": "~$0.13/edit (high quality)",
        },
        "seedream-v4-edit": {
            "id": "fal-ai/bytedance/seedream/v4/edit",
            "name": "Seedream V4 Edit",
            "description": "ByteDance — context-aware image editing with unified architecture",
            "defaults": {},
            "price": "~$0.03/edit",
        },
    },
    "video": {
        "kling-v3-pro": {
            "id": "fal-ai/kling-video/v3/master/text-to-video",
            "name": "Kling V3 Pro",
            "description": "High-quality text/image to video",
            "defaults": {"duration": "5", "aspect_ratio": "16:9"},
            "price": "~$0.30/video",
        },
        "sora-2-t2v": {
            "id": "fal-ai/sora/text-to-video",
            "name": "Sora 2 Text-to-Video",
            "description": "OpenAI Sora text-to-video",
            "defaults": {"duration": 5, "aspect_ratio": "16:9", "resolution": "720p"},
            "price": "~$0.50/video",
        },
        "sora-2-i2v": {
            "id": "fal-ai/sora/image-to-video",
            "name": "Sora 2 Image-to-Video",
            "description": "OpenAI Sora image-to-video",
            "defaults": {"duration": 5, "aspect_ratio": "16:9"},
            "price": "~$0.50/video",
        },
        "ltx-2": {
            "id": "fal-ai/ltx-video/v2",
            "name": "LTX Video V2",
            "description": "Fast, affordable video generation",
            "defaults": {"num_frames": 97, "fps": 24},
            "price": "~$0.05/video",
        },
        "wan-v2": {
            "id": "fal-ai/wan/v2.1/1.3b",
            "name": "Wan 2.1",
            "description": "Alibaba's video generation model",
            "defaults": {"num_frames": 81},
            "price": "~$0.10/video",
        },
    },
    "audio": {
        "chatterbox-tts": {
            "id": "fal-ai/chatterbox/tts",
            "name": "Chatterbox TTS",
            "description": "High-quality text-to-speech",
            "defaults": {},
            "price": "~$0.01/generation",
        },
        "minimax-speech": {
            "id": "fal-ai/minimax/speech-02-hd",
            "name": "MiniMax Speech 02 HD",
            "description": "HD multi-voice speech synthesis",
            "defaults": {},
            "price": "~$0.02/generation",
        },
        "beatoven-music": {
            "id": "fal-ai/beatoven/music",
            "name": "Beatoven Music",
            "description": "AI music generation from text prompt",
            "defaults": {"duration": 30},
            "price": "~$0.05/generation",
        },
        "beatoven-sfx": {
            "id": "fal-ai/beatoven/sfx",
            "name": "Beatoven SFX",
            "description": "Sound effects generation",
            "defaults": {"duration": 10},
            "price": "~$0.03/generation",
        },
    },
    "3d": {
        "meshy-v6": {
            "id": "fal-ai/meshy/image-to-3d/v6",
            "name": "Meshy V6",
            "description": "Image to 3D model generation",
            "defaults": {"topology": "quad", "target_polycount": 30000},
            "price": "~$0.15/model",
        },
        "trellis-2": {
            "id": "fal-ai/trellis/v2",
            "name": "Trellis 2",
            "description": "Fast image to 3D with textures",
            "defaults": {},
            "price": "~$0.10/model",
        },
    },
    "utility": {
        "remove-bg": {
            "id": "fal-ai/bria/background/remove",
            "name": "Background Removal",
            "description": "Remove image background",
            "defaults": {},
            "price": "~$0.005/image",
        },
        "upscale": {
            "id": "fal-ai/aura-sr",
            "name": "Aura SR Upscale",
            "description": "4x image upscaling",
            "defaults": {},
            "price": "~$0.01/image",
        },
    },
}

# Flat lookup: short name → model info
_MODEL_LOOKUP: dict[str, dict[str, Any]] = {}
for _cat, _models in MODEL_CATALOG.items():
    for _key, _info in _models.items():
        _MODEL_LOOKUP[_key] = {**_info, "category": _cat}


def _resolve_model(short_or_full: str, category: str | None = None) -> str:
    """Resolve a short model name to a full fal.ai model ID."""
    if short_or_full.startswith("fal-ai/"):
        return short_or_full
    if short_or_full in _MODEL_LOOKUP:
        return _MODEL_LOOKUP[short_or_full]["id"]
    if category and category in MODEL_CATALOG:
        for key, info in MODEL_CATALOG[category].items():
            if key == short_or_full or info["id"] == short_or_full:
                return info["id"]
    return short_or_full


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOWNLOAD_BASE = Path.home() / "Downloads" / "fal-ai"

MEDIA_TYPE_DIRS = {
    "image": "images",
    "video": "videos",
    "audio": "audio",
    "3d": "3d",
    "other": "other",
}


def _ensure_download_dir(media_type: str) -> Path:
    dirname = MEDIA_TYPE_DIRS.get(media_type, "other")
    path = DOWNLOAD_BASE / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_filename(prefix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(f"{ts}{prefix}{os.urandom(8).hex()}".encode()).hexdigest()[:6]
    return f"{prefix}_{ts}_{h}.{ext.lstrip('.')}"


async def _download_file(url: str, path: Path) -> Path:
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def _guess_extension(url: str, media_type: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path_ext = Path(parsed.path).suffix.lstrip(".")
    if path_ext and len(path_ext) <= 5:
        return path_ext
    defaults = {"image": "png", "video": "mp4", "audio": "wav", "3d": "glb"}
    return defaults.get(media_type, "bin")


async def _download_result_files(
    result: dict, media_type: str, prefix: str
) -> list[dict[str, str]]:
    """Extract URLs from fal.ai result, download them, return file info list."""
    urls: list[str] = []

    if isinstance(result.get("images"), list):
        urls.extend(item.get("url", "") for item in result["images"] if isinstance(item, dict))
    if isinstance(result.get("image"), dict):
        url = result["image"].get("url", "")
        if url:
            urls.append(url)
    if isinstance(result.get("video"), dict):
        url = result["video"].get("url", "")
        if url:
            urls.append(url)
    if isinstance(result.get("audio"), dict):
        url = result["audio"].get("url", "")
        if url:
            urls.append(url)
    if isinstance(result.get("audio_url"), dict):
        url = result["audio_url"].get("url", "")
        if url:
            urls.append(url)
    elif isinstance(result.get("audio_url"), str) and result["audio_url"]:
        urls.append(result["audio_url"])
    if isinstance(result.get("model_mesh"), dict):
        url = result["model_mesh"].get("url", "")
        if url:
            urls.append(url)
    if isinstance(result.get("output"), str) and result["output"].startswith("http"):
        urls.append(result["output"])
    if isinstance(result.get("output"), dict):
        url = result["output"].get("url", "")
        if url:
            urls.append(url)
    if isinstance(result.get("url"), str) and result["url"].startswith("http"):
        urls.append(result["url"])
    if isinstance(result.get("output"), list):
        for item in result["output"]:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])

    seen = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)

    download_dir = _ensure_download_dir(media_type)
    files: list[dict[str, str]] = []

    for url in unique_urls:
        ext = _guess_extension(url, media_type)
        filename = _generate_filename(prefix, ext)
        local_path = download_dir / filename
        try:
            await _download_file(url, local_path)
            files.append({
                "url": url,
                "local_path": str(local_path),
                "filename": filename,
            })
        except Exception:
            files.append({
                "url": url,
                "local_path": "DOWNLOAD_FAILED — use URL directly",
                "filename": filename,
            })

    return files


async def _fal_subscribe(model_id: str, arguments: dict) -> dict:
    result = await fal_client.subscribe_async(model_id, arguments=arguments)
    if isinstance(result, dict):
        return result
    return {"output": result}


def _format_error(e: Exception) -> str:
    msg = str(e)
    if "401" in msg or "Unauthorized" in msg or "unauthorized" in msg:
        return "FAL_KEY is missing or invalid. Check your .claude.json env configuration."
    if "404" in msg or "not found" in msg.lower():
        return f"Model not found. Use list_models to see available models. Error: {msg}"
    if "429" in msg or "rate" in msg.lower():
        return f"Rate limited by fal.ai. Wait a moment and try again. Error: {msg}"
    if "timeout" in msg.lower() or "timed out" in msg.lower():
        return f"Model is busy or overloaded. Try again shortly. Error: {msg}"
    return f"fal.ai error: {msg}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "flux-dev",
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    guidance_scale: float | None = None,
    seed: int | None = None,
) -> dict:
    """Generate images from a text prompt.

    Args:
        prompt: Text description of the image to generate
        model: Model short name or full fal.ai ID. Options: flux-dev (balanced), flux-schnell (fast), flux-pro (best quality), recraft-v4 (design/typography), nano-banana (ultra-fast/cheap), nano-banana-2 (Gemini 3.1 Flash), nano-banana-pro (Gemini 3 Pro, best), gpt-image (GPT-Image 1.5), seedream-v4 (ByteDance, cheap)
        width: Image width in pixels (default 1024). Ignored by nano-banana-2/pro and gpt-image (use resolution/aspect_ratio).
        height: Image height in pixels (default 1024). Ignored by nano-banana-2/pro and gpt-image.
        num_images: Number of images to generate (default 1)
        guidance_scale: How closely to follow the prompt (model-dependent)
        seed: Random seed for reproducibility
    """
    model_id = _resolve_model(model, "image")
    args: dict[str, Any] = {"prompt": prompt}

    # Models that use resolution/aspect_ratio instead of width/height
    if "nano-banana-2" in model_id or "nano-banana-pro" in model_id:
        args["num_images"] = num_images
    elif "gpt-image" in model_id:
        args["image_size"] = f"{width}x{height}"
        args["num_images"] = num_images
        args["quality"] = "high"
    else:
        args["image_size"] = {"width": width, "height": height}
        args["num_images"] = num_images

    if guidance_scale is not None:
        args["guidance_scale"] = guidance_scale
    if seed is not None:
        args["seed"] = seed

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "image", "img")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def edit_image(
    image_url: str,
    prompt: str,
    model: str = "flux-kontext",
    strength: float | None = None,
    seed: int | None = None,
) -> dict:
    """Edit or transform an existing image using AI.

    Args:
        image_url: URL of the source image to edit
        prompt: Description of the desired edit or transformation
        model: Model short name or full fal.ai ID. Options: flux-kontext (default), nano-banana-pro-edit (Gemini 3 Pro), gpt-image-edit (GPT-Image 1.5), seedream-v4-edit (ByteDance, cheap)
        strength: How much to change the image (0.0-1.0, model-dependent)
        seed: Random seed for reproducibility
    """
    # Resolve from both image and image_edit categories
    model_id = _resolve_model(model, "image_edit")
    if model_id == model:
        model_id = _resolve_model(model, "image")
    args: dict[str, Any] = {"prompt": prompt}

    if "kontext" in model_id:
        args["image_url"] = image_url
    elif "nano-banana-pro" in model_id:
        args["image_urls"] = [image_url]
    elif "gpt-image" in model_id:
        args["image_url"] = image_url
        args["quality"] = "high"
    elif "seedream" in model_id:
        args["image_urls"] = [image_url]
    else:
        args["image"] = image_url
        args["image_url"] = image_url

    if strength is not None:
        args["strength"] = strength
    if seed is not None:
        args["seed"] = seed

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "image", "edit")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def generate_video(
    prompt: str = "",
    image_url: str = "",
    model: str = "kling-v3-pro",
    duration: int | str = 5,
    aspect_ratio: str = "16:9",
    seed: int | None = None,
) -> dict:
    """Generate a video from text and/or an image.

    Args:
        prompt: Text description of the video (required for text-to-video)
        image_url: Source image URL (for image-to-video models)
        model: Model short name or full fal.ai ID. Options: kling-v3-pro (best), sora-2-t2v, sora-2-i2v, ltx-2 (fast), wan-v2
        duration: Video duration in seconds (default 5)
        aspect_ratio: Aspect ratio like "16:9", "9:16", "1:1" (default "16:9")
        seed: Random seed for reproducibility
    """
    model_id = _resolve_model(model, "video")
    args: dict[str, Any] = {}

    if prompt:
        args["prompt"] = prompt
    if image_url:
        args["image_url"] = image_url
    args["duration"] = str(duration) if isinstance(duration, int) else duration
    args["aspect_ratio"] = aspect_ratio
    if seed is not None:
        args["seed"] = seed

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "video", "vid")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def generate_audio(
    text: str,
    model: str = "chatterbox-tts",
    voice: str | None = None,
    duration: int | None = None,
) -> dict:
    """Generate audio: speech, music, or sound effects.

    Args:
        text: Text to speak (for TTS) or description of audio to generate (for music/sfx)
        model: Model short name or full fal.ai ID. Options: chatterbox-tts (TTS), minimax-speech (HD TTS), beatoven-music (music), beatoven-sfx (SFX)
        voice: Voice ID for TTS models (model-dependent)
        duration: Duration in seconds (for music/sfx models)
    """
    model_id = _resolve_model(model, "audio")
    args: dict[str, Any] = {}

    if "tts" in model_id or "speech" in model_id:
        args["text"] = text
    else:
        args["prompt"] = text

    if voice is not None:
        args["voice"] = voice
    if duration is not None:
        args["duration"] = duration

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "audio", "audio")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def generate_3d(
    image_url: str,
    model: str = "trellis-2",
    texture_resolution: int | None = None,
) -> dict:
    """Generate a 3D model from an image.

    Args:
        image_url: URL of the source image
        model: Model short name or full fal.ai ID. Options: meshy-v6, trellis-2 (fast)
        texture_resolution: Texture resolution (model-dependent)
    """
    model_id = _resolve_model(model, "3d")
    args: dict[str, Any] = {"image_url": image_url}

    if texture_resolution is not None:
        args["texture_resolution"] = texture_resolution

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "3d", "3d")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def remove_background(image_url: str) -> dict:
    """Remove the background from an image.

    Args:
        image_url: URL of the image to process
    """
    model_id = _resolve_model("remove-bg", "utility")
    args = {"image_url": image_url}

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "image", "nobg")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def upscale_image(image_url: str, scale: int = 4) -> dict:
    """Upscale an image to higher resolution.

    Args:
        image_url: URL of the image to upscale
        scale: Upscale factor (default 4)
    """
    model_id = _resolve_model("upscale", "utility")
    args: dict[str, Any] = {"image_url": image_url}
    if scale != 4:
        args["scale"] = scale

    try:
        result = await _fal_subscribe(model_id, args)
        files = await _download_result_files(result, "image", "upscaled")
        return {"model": model_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def run_model(
    model_id: str,
    arguments: dict,
    media_type: str = "other",
) -> dict:
    """Run any fal.ai model directly with custom arguments.

    Use this for models not covered by the specialized tools, or for advanced parameter control.

    Args:
        model_id: Full fal.ai model ID (e.g. "fal-ai/flux/dev") or short name from catalog
        arguments: Dictionary of arguments to pass to the model
        media_type: Type of output for download organization: "image", "video", "audio", "3d", "other"
    """
    resolved_id = _resolve_model(model_id)

    try:
        result = await _fal_subscribe(resolved_id, arguments)
        files = await _download_result_files(result, media_type, "output")
        return {"model": resolved_id, "files": files, "raw_result": result}
    except Exception as e:
        return {"error": _format_error(e)}


@mcp.tool()
async def list_models(category: str = "") -> dict:
    """List available fal.ai models from the built-in catalog.

    Args:
        category: Filter by category: "image", "image_edit", "video", "audio", "3d", "utility". Empty for all.
    """
    if category and category in MODEL_CATALOG:
        cats = {category: MODEL_CATALOG[category]}
    else:
        cats = MODEL_CATALOG

    result = {}
    for cat_name, models in cats.items():
        result[cat_name] = []
        for short_name, info in models.items():
            result[cat_name].append({
                "short_name": short_name,
                "id": info["id"],
                "name": info["name"],
                "description": info["description"],
                "price": info.get("price", "unknown"),
            })

    return {"models": result, "tip": "Use short_name or full id in model parameters. Use run_model for any fal.ai model not listed here."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
