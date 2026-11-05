from pathlib import Path
import sys

from importlib.metadata import PackageNotFoundError, version as package_version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

project = "M4E 3D"
author = "Alvaro Diaz Flores Caminero"
try:
    release = package_version("M4E")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "build", "api/_generated"]

html_theme = "furo"
html_title = "M4E 3D Documentation"
html_static_path = ["_static"]
pygments_style = "sphinx"
