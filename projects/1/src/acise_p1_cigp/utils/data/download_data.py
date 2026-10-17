import pathlib
import sys
import zipfile
from urllib.parse import unquote, urlparse

import requests

from acise_p1_cigp.utils.logging.log import log

# projects/1/src/acise_p1_cigp/utils/data/download_data.py -> projects/1/data
DATA_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parents[4] / "data"
RAW_DIR: pathlib.Path = DATA_DIR / "raw"

DATA_URLS_FILE: str = "data_urls.txt"

# Only these CSVs are extracted.
ALLOWED_SUFFIX: str = ".csv"
ALLOWED_DIR: str = "circleci_repos/features/"
ALLOWED_FILES: frozenset[str] = frozenset({})

CHUNK_SIZE: int = 1 << 20


def die(msg: str) -> None:
    log.print("[bold red]ERR:[/bold red]", msg)
    sys.exit(2)


def _wanted(name: str) -> bool:
    # Match regardless of any top-level folder prefix inside the archive.
    return name.endswith(ALLOWED_SUFFIX) and (
        pathlib.PurePosixPath(name).name in ALLOWED_FILES or ALLOWED_DIR in name
    )


def _list_contents(zf: zipfile.ZipFile) -> list[str]:
    """Names of the archive members that are safe and needed."""
    return [i.filename for i in zf.infolist() if not i.is_dir() and _wanted(i.filename)]


def _unzip(zf: zipfile.ZipFile, dest: pathlib.Path) -> list[pathlib.Path]:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    extracted: list[pathlib.Path] = []
    for name in _list_contents(zf):
        target = (dest / name).resolve()
        if not target.is_relative_to(dest):  # zip-slip guard
            log.print(f"[yellow]Skipping unsafe path:[/yellow] {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(target, "wb") as out:
            while chunk := src.read(CHUNK_SIZE):
                out.write(chunk)
        extracted.append(target)
    return extracted


def _filename(url: str) -> str:
    name = pathlib.PurePosixPath(unquote(urlparse(url).path)).name
    if not name:
        raise ValueError(f"Cannot derive a file name from {url}")
    return name


def _fetch(session: requests.Session, url: str, out: pathlib.Path) -> None:
    """Stream `url` to `out`; the archive is ~1 GB, so never hold it in memory."""
    tmp = out.with_suffix(out.suffix + ".part")
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with log.status(f"Downloading {out.name}"), open(tmp, "wb") as fout:
            done = 0
            for chunk in resp.iter_content(CHUNK_SIZE):
                fout.write(chunk)
                done += len(chunk)
            log.print(
                f"{out.name}: {done / 1e6:.1f} MB"
                + (f" / {total / 1e6:.1f} MB" if total else "")
            )
    tmp.replace(out)


def download_data(data_urls: list[str]) -> tuple[list[str], list[str]]:
    """Download every URL into DATA_DIR; unpack zips into RAW_DIR (CSVs only).

    Returns (successful urls, failed urls).
    """
    succ, failed = [], []

    session = requests.Session()
    for url in data_urls:
        try:
            out = DATA_DIR / _filename(url)
            if out.exists():
                log.print(f"{out.name} already downloaded, skipping fetch.")
            else:
                _fetch(session, url, out)

            if zipfile.is_zipfile(out):
                with zipfile.ZipFile(out) as zf:
                    files = _unzip(zf, RAW_DIR)
                log.print(f"Extracted {len(files)} CSV files to {RAW_DIR.as_posix()}")
            succ.append(url)
        except Exception:
            log.print(f"[red]Exception encountered while requesting {url}[/red]")
            log.print_exception()
            failed.append(url)

    return succ, failed


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    urls_file = DATA_DIR / DATA_URLS_FILE
    if not urls_file.is_file():
        die(f"{urls_file.as_posix()} not found! Put one URL per line in it.")

    data_urls = [
        u
        for x in urls_file.read_text().splitlines()
        if (u := x.strip()) and not u.startswith("#")
    ]
    if not data_urls:
        die(f"No URLs present in {DATA_URLS_FILE}!")

    succ, failed = download_data(data_urls)
    log.print("Done!", f"Successful: {succ}", f"Failed: {failed}")

    sys.exit(1 if failed else 0)
