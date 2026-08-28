"""
AST-based symbol extractor for staleness anchors in memory storage.

Uses ast-grep-py (PyO3 wrapper around the Rust ast-grep engine) to extract
function, class, method, interface and type names from source files.

Supports: Python, TypeScript, TSX, JavaScript, Go, Rust, Java, Kotlin, C/C++

Install dependency:
    uv pip install ast-grep-py
    # or: pip install ast-grep-py

This module is optional — if ast-grep-py is not installed, extract_symbols()
returns an empty list and remember() proceeds without symbol enrichment.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # TypeScript / TSX
    ".ts": "typescript",
    ".cts": "typescript",
    ".mts": "typescript",
    ".tsx": "tsx",
    # JavaScript / JSX
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}


def _detect_language(file_path: str) -> Optional[str]:
    return EXTENSION_TO_LANGUAGE.get(Path(file_path).suffix.lower())


# ---------------------------------------------------------------------------
# Per-language extraction logic
# ---------------------------------------------------------------------------


def _extract_python(root) -> list[str]:
    symbols: list[str] = []
    for node in root.find_all(kind="class_definition"):
        name = node.field("name")
        if name:
            symbols.append(name.text())
    for node in root.find_all(kind="function_definition"):
        name = node.field("name")
        if name:
            symbols.append(name.text())
    return symbols


def _extract_typescript_js(root, language: str) -> list[str]:
    """Shared extractor for TypeScript, TSX and JavaScript."""
    symbols: list[str] = []

    # Named function declarations: function foo() {}
    for node in root.find_all(kind="function_declaration"):
        name = node.field("name")
        if name:
            symbols.append(name.text())

    # Class declarations
    for node in root.find_all(kind="class_declaration"):
        name = node.field("name")
        if name:
            symbols.append(name.text())

    # Class methods
    for node in root.find_all(kind="method_definition"):
        name = node.field("name")
        if name:
            symbols.append(name.text())

    # Interface declarations (TS/TSX only)
    if language in ("typescript", "tsx"):
        for node in root.find_all(kind="interface_declaration"):
            name = node.field("name")
            if name:
                symbols.append(name.text())

        # Type aliases: type Foo = ...
        for node in root.find_all(kind="type_alias_declaration"):
            name = node.field("name")
            if name:
                symbols.append(name.text())

    # Arrow functions / function expressions assigned to const/let/var:
    #   const foo = () => {}   export const bar = async () => {}
    for node in root.find_all(kind="variable_declarator"):
        value = node.field("value")
        if value and value.kind() in ("arrow_function", "function"):
            name = node.field("name")
            if name:
                symbols.append(name.text())

    return symbols


def _extract_go(root) -> list[str]:
    symbols: list[str] = []
    for node in root.find_all(kind="function_declaration"):
        name = node.field("name")
        if name:
            symbols.append(name.text())
    # Method declarations: func (r Receiver) Method() {}
    for node in root.find_all(kind="method_declaration"):
        name = node.field("name")
        if name:
            symbols.append(name.text())
    for node in root.find_all(kind="type_declaration"):
        for spec in node.find_all(kind="type_spec"):
            name = spec.field("name")
            if name:
                symbols.append(name.text())
    return symbols


def _extract_rust(root) -> list[str]:
    symbols: list[str] = []
    for kind in (
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
    ):
        for node in root.find_all(kind=kind):
            name = node.field("name")
            if name:
                symbols.append(name.text())
    return symbols


def _extract_java(root) -> list[str]:
    symbols: list[str] = []
    for kind in (
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
    ):
        for node in root.find_all(kind=kind):
            name = node.field("name")
            if name:
                symbols.append(name.text())
    return symbols


def _extract_kotlin(root) -> list[str]:
    symbols: list[str] = []
    for kind in ("class_declaration", "object_declaration", "function_declaration"):
        for node in root.find_all(kind=kind):
            name = node.field("name")
            if name:
                symbols.append(name.text())
    return symbols


def _extract_c_cpp(root) -> list[str]:
    symbols: list[str] = []
    for node in root.find_all(kind="function_definition"):
        # C/C++ — name is nested inside declarator
        declarator = node.field("declarator")
        if declarator:
            # May be function_declarator or pointer_declarator wrapping it
            fn_decl = (
                declarator
                if declarator.kind() == "function_declarator"
                else declarator.find(kind="function_declarator")
            )
            if fn_decl:
                name = fn_decl.field("declarator")
                if name:
                    symbols.append(name.text())
    for node in root.find_all(kind="struct_specifier"):
        name = node.field("name")
        if name:
            symbols.append(name.text())
    return symbols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_symbols(file_path: str) -> list[str]:
    """
    Extract symbol names (functions, classes, methods, interfaces, types)
    from a single source file using ast-grep-py.

    Returns a sorted, deduplicated list of symbol name strings.
    Returns [] if the file is unsupported, unreadable, or ast-grep-py is
    not installed.
    """
    try:
        from ast_grep_py import SgRoot  # type: ignore[import]
    except ImportError:
        logger.debug("ast-grep-py not installed — symbol extraction skipped")
        return []

    path = Path(file_path)
    language = _detect_language(file_path)
    if not language:
        logger.debug("Unsupported extension for AST extraction: %s", path.suffix)
        return []

    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        root = SgRoot(src, language).root()

        if language == "python":
            symbols = _extract_python(root)
        elif language in ("typescript", "tsx", "javascript"):
            symbols = _extract_typescript_js(root, language)
        elif language == "go":
            symbols = _extract_go(root)
        elif language == "rust":
            symbols = _extract_rust(root)
        elif language == "java":
            symbols = _extract_java(root)
        elif language == "kotlin":
            symbols = _extract_kotlin(root)
        elif language in ("c", "cpp"):
            symbols = _extract_c_cpp(root)
        else:
            symbols = []

        return sorted(set(s for s in symbols if s))

    except Exception as exc:
        logger.warning("AST extraction failed for %s: %s", file_path, exc)
        return []


def extract_symbols_multi(file_paths: list[str]) -> dict[str, list[str]]:
    """
    Extract symbols from multiple files.

    Returns a dict mapping filename (basename only) → [symbol_names].
    Files that yield no symbols are omitted from the result.
    """
    result: dict[str, list[str]] = {}
    for fp in file_paths:
        syms = extract_symbols(fp)
        if syms:
            result[Path(fp).name] = syms
    return result


# ---------------------------------------------------------------------------
# File content hash
# ---------------------------------------------------------------------------


def hash_file(file_path: str) -> Optional[str]:
    """
    Return a short SHA-256 hex digest (first 16 chars) of the file's content.

    Used as a staleness anchor — if the hash changes, the file was modified
    since the memory was stored, regardless of which symbols changed.

    Returns None if the file is unreadable.
    """
    try:
        content = Path(file_path).read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception as exc:
        logger.warning("hash_file failed for %s: %s", file_path, exc)
        return None


def hash_files(file_paths: list[str]) -> dict[str, str]:
    """
    Hash multiple files.

    Returns a dict mapping basename → short SHA-256 hex digest.
    Files that cannot be read are omitted.
    """
    result: dict[str, str] = {}
    for fp in file_paths:
        h = hash_file(fp)
        if h is not None:
            result[Path(fp).name] = h
    return result


# ---------------------------------------------------------------------------
# Function signature extraction
# ---------------------------------------------------------------------------


def _extract_python_signatures(root) -> dict[str, str]:
    """
    Extract function/method signatures from a Python AST root.

    Returns a dict mapping function_name → short hash of its parameter list text.
    This detects parameter additions, removals and renames without storing raw source.
    """
    sigs: dict[str, str] = {}
    for node in root.find_all(kind="function_definition"):
        name_node = node.field("name")
        params_node = node.field("parameters")
        if name_node and params_node:
            name = name_node.text()
            param_text = params_node.text()
            sigs[name] = hashlib.sha256(param_text.encode()).hexdigest()[:8]
    return sigs


def _extract_ts_js_signatures(root, language: str) -> dict[str, str]:
    """Extract function/method signatures from TypeScript/JavaScript."""
    sigs: dict[str, str] = {}

    def _add(node, name_field: str = "name", params_field: str = "parameters"):
        name_node = node.field(name_field)
        params_node = node.field(params_field)
        if name_node and params_node:
            sigs[name_node.text()] = hashlib.sha256(
                params_node.text().encode()
            ).hexdigest()[:8]

    for node in root.find_all(kind="function_declaration"):
        _add(node)
    for node in root.find_all(kind="method_definition"):
        _add(node)
    # Arrow functions assigned to variables
    for node in root.find_all(kind="variable_declarator"):
        value = node.field("value")
        if value and value.kind() in ("arrow_function", "function"):
            name_node = node.field("name")
            params_node = value.field("parameters")
            if name_node and params_node:
                sigs[name_node.text()] = hashlib.sha256(
                    params_node.text().encode()
                ).hexdigest()[:8]
    return sigs


def extract_signatures(file_path: str) -> dict[str, str]:
    """
    Extract a signature-hash map from a single source file.

    Returns {function_name: param_hash} where param_hash is an 8-char
    SHA-256 digest of the parameter list text.

    If the param hash for a function changes between storage time and recall
    time, the function's interface changed — a strong staleness signal even
    when the symbol name is unchanged.

    Currently supports Python, TypeScript, TSX, JavaScript.
    Returns {} for unsupported languages or if ast-grep-py is not installed.
    """
    try:
        from ast_grep_py import SgRoot  # type: ignore[import]
    except ImportError:
        return {}

    path = Path(file_path)
    language = _detect_language(file_path)
    if not language:
        return {}

    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        root = SgRoot(src, language).root()
        if language == "python":
            return _extract_python_signatures(root)
        elif language in ("typescript", "tsx", "javascript"):
            return _extract_ts_js_signatures(root, language)
        else:
            return {}
    except Exception as exc:
        logger.warning("extract_signatures failed for %s: %s", file_path, exc)
        return {}


def extract_signatures_multi(file_paths: list[str]) -> dict[str, dict[str, str]]:
    """
    Extract signatures from multiple files.

    Returns {basename: {func_name: param_hash}}.
    Files with no signatures are omitted.
    """
    result: dict[str, dict[str, str]] = {}
    for fp in file_paths:
        sigs = extract_signatures(fp)
        if sigs:
            result[Path(fp).name] = sigs
    return result


# ---------------------------------------------------------------------------
# Git commit anchor
# ---------------------------------------------------------------------------


def get_git_commit(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Return the current git HEAD short commit hash (7 chars).

    Runs `git rev-parse --short HEAD` in repo_path (or cwd if None).
    Returns None if git is not available or the directory is not a repo.

    Used as a staleness anchor — after storage, the agent can run
    `git log --oneline <stored_hash>..HEAD -- <file>` to see exactly
    which commits touched the referenced files since the memory was written.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as exc:
        logger.debug("get_git_commit failed: %s", exc)
        return None
