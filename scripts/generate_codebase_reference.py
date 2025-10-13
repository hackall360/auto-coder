"""Generate a Markdown reference covering the core Auto-Coder modules.

This helper walks the repository tree, extracts module, class, and function
docstrings via the :mod:`ast` module, and renders a compact documentation page
summarising the public surface area.  Test modules are intentionally skipped to
keep the output focused on the production runtime.  Regenerate the reference
whenever you add, rename, or remove modules so contributors always have an
up-to-date map of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "autocoder" / "codebase-reference.md"


EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "assets",
    "docs",
    "plans",
    "tests",
}


@dataclass(slots=True)
class MethodInfo:
    """Docstring metadata for a function or method."""

    name: str
    docstring: str = ""


@dataclass(slots=True)
class ClassInfo:
    """Docstring metadata for a class and its public methods."""

    name: str
    docstring: str = ""
    methods: list[MethodInfo] = field(default_factory=list)


@dataclass(slots=True)
class ModuleInfo:
    """Aggregated documentation for a module."""

    path: Path
    docstring: str = ""
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[MethodInfo] = field(default_factory=list)


def iter_source_files(root: Path) -> list[Path]:
    """Return Python source files excluding test/docs/support directories."""

    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def trim_docstring(text: str) -> str:
    """Normalise indentation and collapse docstrings to paragraphs."""

    if not text:
        return ""
    cleaned = textwrap.dedent(text).strip()
    if not cleaned:
        return ""
    # Keep the first two paragraphs to avoid overwhelming the summary.
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        return "\n\n".join(paragraphs)
    return "\n\n".join(paragraphs[:2]) + "\n\n…"


def collect_module_info(path: Path) -> ModuleInfo:
    """Parse a Python module and return docstring metadata."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_doc = trim_docstring(ast.get_docstring(tree) or "")
    classes: list[ClassInfo] = []
    functions: list[MethodInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            class_doc = trim_docstring(ast.get_docstring(node) or "")
            methods: list[MethodInfo] = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                    method_doc = trim_docstring(ast.get_docstring(child) or "")
                    methods.append(MethodInfo(child.name, method_doc))
            classes.append(ClassInfo(node.name, class_doc, methods))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            func_doc = trim_docstring(ast.get_docstring(node) or "")
            functions.append(MethodInfo(node.name, func_doc))

    return ModuleInfo(path=path.relative_to(REPO_ROOT), docstring=module_doc, classes=classes, functions=functions)


def render_module_section(info: ModuleInfo) -> str:
    """Render a Markdown section for a single module."""

    lines: list[str] = []
    lines.append(f"### `{info.path}`")
    lines.append("")
    if info.docstring:
        lines.append(info.docstring)
        lines.append("")
    else:
        lines.append("_No module-level docstring._")
        lines.append("")

    if info.classes:
        lines.append("#### Public classes")
        lines.append("")
        for cls in info.classes:
            lines.append(f"- **{cls.name}**")
            if cls.docstring:
                lines.append("")
                lines.append(textwrap.indent(cls.docstring, "    "))
                lines.append("")
            if cls.methods:
                lines.append("    - Methods:")
                for method in cls.methods:
                    summary = method.docstring.splitlines()[0] if method.docstring else ""
                    if summary:
                        lines.append(f"        - `{method.name}()` – {summary}")
                    else:
                        lines.append(f"        - `{method.name}()`")
                lines.append("")
    if info.functions:
        lines.append("#### Top-level functions")
        lines.append("")
        for func in info.functions:
            summary = func.docstring.splitlines()[0] if func.docstring else ""
            if summary:
                lines.append(f"- `{func.name}()` – {summary}")
            else:
                lines.append(f"- `{func.name}()`")
        lines.append("")

    return "\n".join(line for line in lines if line != "") + "\n"


def build_documentation() -> str:
    """Generate the full Markdown document."""

    modules = [collect_module_info(path) for path in iter_source_files(REPO_ROOT)]
    modules.sort(key=lambda info: (info.path.parent.parts, info.path.name))
    header = textwrap.dedent(
        """
        # Auto-Coder Codebase Reference

        _This file is auto-generated by `scripts/generate_codebase_reference.py`. Run the
        script after modifying any modules to keep the inventory current._
        """
    ).strip()

    sections = [header, ""]
    current_dir: Path | None = None

    for module in modules:
        parent = module.path.parent
        if current_dir is None or parent != current_dir:
            current_dir = parent
            if parent == Path("."):
                sections.append("## Top-level modules")
            else:
                sections.append(f"## `{parent}/`")
            sections.append("")
        sections.append(render_module_section(module))

    return "\n".join(sections).strip() + "\n"


def main() -> None:
    """Write the generated documentation to ``docs/autocoder/codebase-reference.md``."""

    content = build_documentation()
    DOC_PATH.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote documentation for {DOC_PATH}")


if __name__ == "__main__":
    main()
