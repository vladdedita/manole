"""
Preprocessing step: use LFM2.5-VL to generate text descriptions for images
and scanned PDFs before indexing with LEANN.

Creates sidecar .description.txt files next to each processed file.
LEANN's `leann build` will then pick up both originals and descriptions.

Usage:
    uv run python preprocess.py /path/to/directory
    uv run python preprocess.py                      # defaults to ./test_data
    uv run python preprocess.py --force               # regenerate existing descriptions
"""

import sys
import time
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
DESCRIPTION_SUFFIX = ".description.txt"

# Prompts for the VL model
IMAGE_PROMPT = (
    "Describe this image in detail. Include: what objects are visible, "
    "any text or writing, colors, layout, and the overall subject matter. "
    "If it's a diagram, describe its structure and labels. "
    "If it contains text, transcribe it exactly."
)
PDF_PAGE_PROMPT = (
    "This is a scanned document page. Extract and transcribe ALL text you can see. "
    "Preserve the structure (headings, paragraphs, lists). "
    "If there are images, diagrams, or tables, describe them too."
)


def load_vl_model():
    """Load LFM2.5-VL model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "LiquidAI/LFM2.5-VL-1.6B"
    print(f"Loading {model_id}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def describe_image(model, processor, image: Image.Image, prompt: str) -> str:
    """Generate a text description of an image using LFM2.5-VL."""
    conversation = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract assistant response
    if "assistant" in result:
        result = result.split("assistant")[-1].strip()
    return result


def process_image_file(model, processor, path: Path) -> str:
    """Process a single image file."""
    img = Image.open(path).convert("RGB")
    return describe_image(model, processor, img, IMAGE_PROMPT)


def process_pdf_file(model, processor, path: Path) -> str:
    """Process a PDF — first try text extraction, fall back to VL for scanned pages."""
    import pymupdf

    doc = pymupdf.open(str(path))
    page_texts = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) > 50:
            # Page has enough text, use it directly
            page_texts.append(f"--- Page {i + 1} ---\n{text}")
        else:
            # Scanned page or very little text — use VL model
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            description = describe_image(model, processor, img, PDF_PAGE_PROMPT)
            page_texts.append(f"--- Page {i + 1} (scanned) ---\n{description}")

    doc.close()
    return "\n\n".join(page_texts)


def discover_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Find image and PDF files that need processing."""
    images = []
    pdfs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(path)
        elif ext in PDF_EXTENSIONS:
            pdfs.append(path)
    return images, pdfs


def needs_processing(path: Path, force: bool) -> bool:
    """Check if a file needs a description generated."""
    desc_path = path.parent / (path.name + DESCRIPTION_SUFFIX)
    if force:
        return True
    return not desc_path.exists()


def write_description(path: Path, description: str, file_type: str):
    """Write a sidecar description file."""
    desc_path = path.parent / (path.name + DESCRIPTION_SUFFIX)
    header = (
        f"[Auto-generated description of {path.name}]\n"
        f"Source: {path.name}\n"
        f"Type: {file_type}\n"
        f"---\n"
    )
    desc_path.write_text(header + description, encoding="utf-8")
    return desc_path


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]

    data_dir = Path(args[0]).resolve() if args else Path("./test_data").resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    images, pdfs = discover_files(data_dir)
    to_process_images = [f for f in images if needs_processing(f, force)]
    to_process_pdfs = [f for f in pdfs if needs_processing(f, force)]

    total = len(to_process_images) + len(to_process_pdfs)

    print(f"Directory: {data_dir}")
    print(f"Found: {len(images)} images, {len(pdfs)} PDFs")
    print(f"Need processing: {len(to_process_images)} images, {len(to_process_pdfs)} PDFs")

    if total == 0:
        print("Nothing to process. Use --force to regenerate.")
        return

    model, processor = load_vl_model()

    for i, path in enumerate(to_process_images, 1):
        print(f"\n[{i}/{total}] Image: {path.relative_to(data_dir)}")
        t0 = time.time()
        description = process_image_file(model, processor, path)
        desc_path = write_description(path, description, "image")
        print(f"  -> {desc_path.name} ({time.time() - t0:.1f}s)")
        print(f"  Preview: {description[:120]}...")

    for i, path in enumerate(to_process_pdfs, len(to_process_images) + 1):
        print(f"\n[{i}/{total}] PDF: {path.relative_to(data_dir)}")
        t0 = time.time()
        description = process_pdf_file(model, processor, path)
        desc_path = write_description(path, description, "pdf-ocr")
        print(f"  -> {desc_path.name} ({time.time() - t0:.1f}s)")
        print(f"  Preview: {description[:120]}...")

    print(f"\nDone. {total} files processed.")
    print("Run `uv run python chat.py <dir>` to index and chat.")


if __name__ == "__main__":
    main()
