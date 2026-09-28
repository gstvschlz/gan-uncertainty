"""Verify the Pages site (`mise run docs`): docs/index.html exists, every relative
src/href resolves inside docs/, the four GIFs exist, and docs/ stays within 5 MB."""

from html.parser import HTMLParser
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
GIFS = ["ti-latent-walk", "ti-catalog", "training-progression", "realizations"]
MAX_BYTES = 5 * 1024 * 1024


class Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.refs += [v for k, v in attrs if k in ("src", "href") and v]


index = DOCS / "index.html"
assert index.exists(), f"{index} not found"
parser = Refs()
parser.feed(index.read_text(encoding="utf-8"))

for ref in parser.refs:
    if ref.startswith(("http://", "https://", "mailto:", "#")):
        continue
    path = ref.split("#")[0].split("?")[0]
    assert (DOCS / path).is_file(), f"index.html points to missing file: {ref}"
for name in GIFS:
    assert (DOCS / f"{name}.gif").is_file(), f"missing docs/{name}.gif"

size = sum(f.stat().st_size for f in DOCS.rglob("*") if f.is_file())
assert size <= MAX_BYTES, f"docs/ is {size / 1e6:.2f} MB, limit 5 MB"
print(f"OK: {len(parser.refs)} references checked, docs/ is {size / 1e6:.2f} MB")
print(f"Preview: {index.as_uri()}")
