import time

import pypdfium2  # Needs to be at the top to avoid warnings
import os
import fire

# For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models

from marker.output import save_markdown
from marker.settings import settings

settings.EXTRACT_IMAGES = False

configure_logging()


def main(
    filename: str,
    output: str,
    max_pages: int | None = None,
    start_page: int | None = None,
    langs: str | None = None,
    batch_multiplier: int = 2,
    debug: bool = False,
):
    """Convert single PDF file to markdown format

    Args:
        filename (str): PDF file to parse
        output (str): Output base folder path
        max_pages (int): Maximum number of pages to parse
        start_page (int): Page to start processing at
        langs (str): Languages to use for OCR, comma separated
        batch_multiplier (int, optional): How much to increase batch sizes. Defaults to 2.
        debug (bool, optional): Enable debug logging. Defaults to False.
    """

    langs = langs.split(",") if langs else None

    fname = filename
    model_lst = load_all_models()
    start = time.time()
    full_text, images, out_meta = convert_single_pdf(
        fname,
        model_lst,
        max_pages=max_pages,
        langs=langs,
        batch_multiplier=batch_multiplier,
        start_page=start_page,
    )

    fname = os.path.basename(fname)
    subfolder_path = save_markdown(output, fname, full_text, images, out_meta)

    print(f"Saved markdown to the {subfolder_path} folder")
    if debug:
        print(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    fire.Fire(main)
