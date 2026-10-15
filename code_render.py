#!/usr/bin/env python3
"""Generate SVG code with an OpenAI-compatible VLM API and render it to PNG.

The output layout and result JSON intentionally match the rest of UniBench:

    results/<model_name>/<data_name>/<index>.svg
    results/<model_name>/<data_name>/<index>.png
    results/<data_name>_<model_name>.json

Example:

    export OPENAI_API_KEY=...
    python model_eval/code_render.py \
        --data-name ui_gen \
        --limit 3

The model only sees the benchmark prompt (and the original image for editing
tasks). Correctness key points and reference images are deliberately not sent.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "travel_plan_gen": "map/generation/map_gen.json",
    "travel_plan_edit": "map/editing/map_editing_prompts.json",
    "process_plan_gen": "diagram/generation/diagram_gen.json",
    "process_plan_edit": "diagram/editing/diagram_edit.json",
    "ui_gen": "UI/generation/ui_generation.json",
    "ui_gen_diff": "UI/generation/ui_generation_diff.json",
    "ui_edit": "UI/editing/ui_edit.json",
    "random_style": "random_style.json",
}

SVG_OPEN_RE = re.compile(r"<svg\b", re.IGNORECASE)
SVG_CLOSE_RE = re.compile(r"</svg\s*>", re.IGNORECASE)
EXTERNAL_URL_RE = re.compile(r"(?:https?://|file://|javascript:|@import)", re.IGNORECASE)
PLACEHOLDER = "__INPUT_IMAGE__"

SYSTEM_PROMPT = """You are an expert visual designer and SVG programmer.
Create the requested final image as SVG code. Return code only: exactly one
standalone <svg>...</svg> document, with no Markdown fence and no explanation.

Rules:
- Use only standard SVG elements and inline CSS. Never use JavaScript,
  foreignObject, external URLs, external fonts, or external files.
- Make all important text readable and meaningful. Use <text>/<tspan> elements.
- Use a clean visual hierarchy, good spacing, high contrast, and clear arrows.
- The result must be complete and understandable without interaction.
- Keep every visible element inside the viewBox.
"""


@dataclass
class SampleResult:
    index: int
    success: bool
    elapsed_seconds: float
    attempts: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible VLM -> SVG code -> PNG baseline for UniBench"
    )
    parser.add_argument("--data-name", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model",
        default=os.getenv("CODE_RENDER_MODEL", "gpt-5.6-sol"),
        help="Model identifier sent to the API",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.6-sol-svg",
        help="Filesystem/result name for this baseline",
    )
    parser.add_argument(
        "--api-style",
        choices=("responses", "chat-completions"),
        default=os.getenv("CODE_RENDER_API_STYLE", "responses"),
        help="OpenAI Responses API by default; use chat-completions for compatible providers",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "CODE_RENDER_BASE_URL",
            "https://api.openai.com/v1",
        ),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument("--start", type=int, default=0, help="First dataset index")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API requests")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum combined reasoning and visible output tokens",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "original", "auto"),
        default="original",
        help="Vision detail for editing inputs",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature; incompatible with non-none reasoning",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=1024,
        help="Generation canvas size and maximum editing-image edge",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate samples that already have both SVG and PNG",
    )
    parser.add_argument(
        "--no-original-layer",
        action="store_true",
        help="Do not allow edit SVGs to embed the original image as a base layer",
    )
    return parser.parse_args()


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        with Image.open(path) as image:
            fmt = (image.format or "PNG").lower()
        mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def canvas_size(sample: dict[str, Any], output_size: int) -> tuple[int, int]:
    image_path = sample.get("image_path")
    if not image_path:
        return output_size, output_size

    source = REPO_ROOT / image_path
    with Image.open(source) as image:
        width, height = image.size
    scale = output_size / max(width, height)
    scaled_width = max(8, int(round(width * scale / 8) * 8))
    scaled_height = max(8, int(round(height * scale / 8) * 8))
    return scaled_width, scaled_height


def build_prompt(
    sample: dict[str, Any],
    data_name: str,
    width: int,
    height: int,
    allow_original_layer: bool,
    previous_error: str | None = None,
) -> str:
    is_edit = bool(sample.get("image_path"))
    task_family = data_name.split("_", 1)[0]
    family_guidance = {
        "travel": (
            "This is a map/travel task. Make routes visually continuous and clearly marked, "
            "use arrows or numbered stops, and label the requested real places."
        ),
        "process": (
            "This is a process-diagram task. Use labeled nodes and unambiguous directed "
            "connectors. Ensure the order and branching logic are easy to follow."
        ),
        "ui": (
            "This is a UI task. Produce a polished static interface mockup with readable "
            "labels, aligned controls, and a clear state matching the request."
        ),
        "random": "Choose the most appropriate structured visual design for the request.",
    }.get(task_family, "Create a clear structured visual design.")

    lines = [
        f"User request: {sample['prompt']}",
        family_guidance,
        f"Use a canvas of {width} by {height} with viewBox='0 0 {width} {height}'.",
        "Set width and height on the root SVG and include xmlns='http://www.w3.org/2000/svg'.",
    ]

    if is_edit:
        lines.append(
            "The supplied image is the original image. Return the final image after applying "
            "the requested change, not an explanation of the change."
        )
        if allow_original_layer:
            lines.append(
                f"When preservation is useful, include the original as a base layer exactly as "
                f"<image href='{PLACEHOLDER}' x='0' y='0' width='{width}' height='{height}'/> "
                "and draw SVG changes above it. You may instead redraw the full result when that "
                "better satisfies the request."
            )
        else:
            lines.append("Redraw the complete final result in SVG; do not embed the original image.")
    else:
        lines.append("Create the complete image from scratch in SVG.")

    if previous_error:
        lines.append(
            "Your previous SVG could not be accepted. Produce a fresh, valid SVG and correct "
            f"this problem: {previous_error[:500]}"
        )
    return "\n".join(lines)


def extract_svg(response_text: str) -> str:
    start_match = SVG_OPEN_RE.search(response_text)
    if not start_match:
        raise ValueError("API response did not contain an <svg> element")
    close_matches = list(SVG_CLOSE_RE.finditer(response_text, start_match.start()))
    if not close_matches:
        raise ValueError("API response did not contain a closing </svg> tag")
    svg = response_text[start_match.start() : close_matches[-1].end()].strip()
    if "xmlns=" not in svg[:500]:
        svg = re.sub(
            r"<svg\b",
            '<svg xmlns="http://www.w3.org/2000/svg"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )
    if "xlink:" in svg and "xmlns:xlink=" not in svg[:800]:
        svg = re.sub(
            r"<svg\b",
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )
    return svg


def validate_and_normalize_svg(
    svg: str,
    width: int,
    height: int,
    original_data_uri: str | None,
) -> str:
    if re.search(r"<!DOCTYPE|<!ENTITY", svg, re.IGNORECASE):
        raise ValueError("DOCTYPE and ENTITY declarations are forbidden")

    if PLACEHOLDER in svg:
        if original_data_uri is None:
            raise ValueError("SVG used the input-image placeholder in a generation task")
        svg = svg.replace(PLACEHOLDER, original_data_uri)

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise ValueError(f"invalid XML: {exc}") from exc

    root_name = root.tag.rsplit("}", 1)[-1].lower()
    if root_name != "svg":
        raise ValueError("root XML element is not <svg>")

    forbidden_tags = {"script", "foreignobject", "iframe", "audio", "video"}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in forbidden_tags:
            raise ValueError(f"forbidden SVG element: <{tag}>")
        if element.text and EXTERNAL_URL_RE.search(element.text):
            raise ValueError("external URL or import found in SVG text/style")
        for attribute, value in element.attrib.items():
            attr_name = attribute.rsplit("}", 1)[-1].lower()
            value = value.strip()
            if attr_name.startswith("on"):
                raise ValueError(f"event handler attribute is forbidden: {attr_name}")
            if EXTERNAL_URL_RE.search(value):
                raise ValueError(f"external URL is forbidden in attribute: {attr_name}")
            if attr_name == "href" and value and not (
                value.startswith("#") or value.startswith("data:image/")
            ):
                raise ValueError(f"unsupported href in SVG: {value[:80]}")

    root.set("width", str(width))
    root.set("height", str(height))
    if not any(key.lower() == "viewbox" for key in root.attrib):
        root.set("viewBox", f"0 0 {width} {height}")
    return ET.tostring(root, encoding="unicode")


def render_svg(svg: str, svg_path: Path, png_path: Path, width: int, height: int) -> None:
    svg_path.write_text(svg, encoding="utf-8")
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=width,
            output_height=height,
        )
    except ImportError:
        executable = shutil.which("cairosvg")
        if not executable:
            raise RuntimeError(
                "CairoSVG is unavailable. Install cairosvg or expose its CLI on PATH."
            )
        completed = subprocess.run(
            [
                executable,
                str(svg_path),
                "-f",
                "png",
                "-o",
                str(png_path),
                "--output-width",
                str(width),
                "--output-height",
                str(height),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"CairoSVG failed: {completed.stderr.strip()[:500]}")

    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError("renderer did not produce a non-empty PNG")
    with Image.open(png_path) as rendered:
        rendered.verify()


def create_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai Python package is required") from exc

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API key found. Export {args.api_key_env} (or OPENAI_API_KEY)."
        )
    return OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)


def request_svg_code(
    client: Any,
    args: argparse.Namespace,
    sample: dict[str, Any],
    data_name: str,
    width: int,
    height: int,
    previous_error: str | None,
) -> str:
    prompt = build_prompt(
        sample,
        data_name,
        width,
        height,
        not args.no_original_layer,
        previous_error,
    )
    image_path = sample.get("image_path")
    source_data_uri = image_data_uri(REPO_ROOT / image_path) if image_path else None

    if args.api_style == "responses":
        content: list[dict[str, Any]] = []
        if source_data_uri:
            content.append(
                {
                    "type": "input_image",
                    "image_url": source_data_uri,
                    "detail": args.image_detail,
                }
            )
        content.append({"type": "input_text", "text": prompt})
        request: dict[str, Any] = {
            "model": args.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": args.reasoning_effort},
            "max_output_tokens": args.max_tokens,
            "store": False,
        }
        if args.temperature is not None:
            request["temperature"] = args.temperature
        response = client.responses.create(**request)
        response_text = response.output_text
    else:
        content = []
        if source_data_uri:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": source_data_uri,
                        "detail": args.image_detail,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": args.max_tokens,
        }
        if args.reasoning_effort != "none":
            request["reasoning_effort"] = args.reasoning_effort
        if args.temperature is not None:
            request["temperature"] = args.temperature
        completion = client.chat.completions.create(**request)
        response_text = completion.choices[0].message.content
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("API returned empty content")
    return response_text


def process_sample(
    index: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
    output_dir: Path,
) -> SampleResult:
    started = time.monotonic()
    svg_path = output_dir / f"{index}.svg"
    png_path = output_dir / f"{index}.png"
    raw_path = output_dir / f"{index}.response.txt"
    error_path = output_dir / f"{index}.error.txt"

    if (
        not args.overwrite
        and svg_path.is_file()
        and png_path.is_file()
        and not error_path.exists()
    ):
        return SampleResult(index, True, 0.0, 0)

    width, height = canvas_size(sample, args.output_size)
    original_data_uri = None
    if sample.get("image_path") and not args.no_original_layer:
        original_data_uri = image_data_uri(REPO_ROOT / sample["image_path"])

    previous_error = None
    last_response = ""
    for attempt in range(1, args.retries + 1):
        try:
            last_response = request_svg_code(
                client, args, sample, args.data_name, width, height, previous_error
            )
            raw_path.write_text(last_response, encoding="utf-8")
            svg = extract_svg(last_response)
            svg = validate_and_normalize_svg(
                svg, width, height, original_data_uri
            )
            render_svg(svg, svg_path, png_path, width, height)
            error_path.unlink(missing_ok=True)
            return SampleResult(
                index=index,
                success=True,
                elapsed_seconds=time.monotonic() - started,
                attempts=attempt,
            )
        except Exception as exc:  # Preserve the failure and retry this sample.
            previous_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    error_path.write_text(previous_error or "unknown error", encoding="utf-8")
    return SampleResult(
        index=index,
        success=False,
        elapsed_seconds=time.monotonic() - started,
        attempts=args.retries,
        error=previous_error,
    )


def main() -> int:
    args = parse_args()
    if args.start < 0 or args.limit is not None and args.limit < 1:
        raise ValueError("--start must be non-negative and --limit must be positive")
    if args.workers < 1 or args.retries < 1:
        raise ValueError("--workers and --retries must be positive")
    if args.output_size < 64:
        raise ValueError("--output-size must be at least 64")
    if args.temperature is not None and args.reasoning_effort != "none":
        raise ValueError(
            "--temperature is only supported when --reasoning-effort=none"
        )
    if "/" in args.model_name or "\\" in args.model_name:
        raise ValueError("--model-name must not contain path separators")

    dataset_path = REPO_ROOT / DATASETS[args.data_name]
    with dataset_path.open("r", encoding="utf-8") as handle:
        source_data = json.load(handle)
    if not isinstance(source_data, list):
        raise ValueError(f"dataset is not a JSON list: {dataset_path}")

    end = len(source_data) if args.limit is None else min(
        len(source_data), args.start + args.limit
    )
    indices = list(range(min(args.start, len(source_data)), end))
    if not indices:
        print("No samples selected.", file=sys.stderr)
        return 2

    output_dir = REPO_ROOT / "results" / args.model_name / args.data_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = REPO_ROOT / "results" / f"{args.data_name}_{args.model_name}.json"

    results_data = copy.deepcopy(source_data)
    if result_json.is_file():
        try:
            existing = json.loads(result_json.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) == len(results_data):
                results_data = existing
        except (OSError, json.JSONDecodeError):
            pass

    client = create_client(args)
    print(
        f"Running {args.model} ({args.api_style}, reasoning={args.reasoning_effort}) "
        f"on {args.data_name}: samples {indices[0]}..{indices[-1]}, "
        f"workers={args.workers}"
    )

    lock = threading.Lock()
    succeeded = 0
    failed = 0

    def finish(result: SampleResult) -> None:
        nonlocal succeeded, failed
        line = results_data[result.index]
        line["code_render_model"] = args.model
        line["code_render_api_style"] = args.api_style
        line["code_render_reasoning_effort"] = args.reasoning_effort
        line["code_render_attempts"] = result.attempts
        line["code_render_seconds"] = round(result.elapsed_seconds, 3)
        if result.success:
            svg_path = output_dir / f"{result.index}.svg"
            png_path = output_dir / f"{result.index}.png"
            line["code_path"] = relative_to_repo(svg_path)
            line["res_path"] = relative_to_repo(png_path)
            line["render_status"] = "success"
            line.pop("render_error", None)
            succeeded += 1
            status = "OK"
        else:
            line.pop("code_path", None)
            line.pop("res_path", None)
            line["render_status"] = "failed"
            line["render_error"] = result.error
            failed += 1
            status = "FAILED"
        atomic_json_dump(results_data, result_json)
        print(
            f"[{result.index + 1}/{len(source_data)}] {status} "
            f"attempts={result.attempts} time={result.elapsed_seconds:.1f}s"
            + (f" error={result.error}" if result.error else ""),
            flush=True,
        )

    if args.workers == 1:
        for index in indices:
            finish(process_sample(index, source_data[index], args, client, output_dir))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_sample,
                    index,
                    source_data[index],
                    args,
                    client,
                    output_dir,
                ): index
                for index in indices
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    finish(result)

    print(f"Finished: success={succeeded}, failed={failed}")
    print(f"Result JSON: {result_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
