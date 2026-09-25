#!/usr/bin/env python3
"""Enforce the project's central architectural invariant.

    Agent autonomy is scoped by the reversibility of the action space.

The agents under `agents/` reason autonomously over read-only tools. They never
perform an irreversible operation -- not a database write, not an external
booking call, not even a 120-second Redis hold. Every such operation lives
behind the deterministic validate -> execute -> verify path in `graph/nodes/`,
which the model cannot reach.

This script turns that claim into a check that fails the build, rather than a
promise in a README. It is deliberately written before the code it guards:
a rule added after the first violation is a rule nobody will enforce.

Two rules, because a write can hide in two ways:

  IMPORTS   agents/ may not import the modules that can write -- the database
            layer, the calendar adapters, Redis, the job queue. Agents reach
            data only through the read-only tools in tools/. This is the
            structural rule: it does not need to know any method names.
  PATTERNS  write-shaped calls and raw SQL (below), as a second net for
            anything the import rule cannot see.

Precision matters more than coverage here. A noisy check gets commented out, so
patterns are matched against code with comments and string literals stripped,
and `# noqa: invariant -- <reason>` documents a reviewed exception.

Usage:  python scripts/check_invariants.py [--verbose]
Exit:   0 = clean, 1 = violations found
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "api" / "src"
GUARDED_DIR = SRC_ROOT / "patient_ops" / "agents"

ALLOW_MARKER = "noqa: invariant"

# Modules agents/ must not import -- each one can change state, or is a layer
# agents are not allowed to reach into directly. A module and everything
# beneath it: "patient_ops.db" also covers "patient_ops.db.repo".
# Grows with the project: Phase 10 adds patient_ops.tools.escalation.
DENIED_IMPORTS: tuple[str, ...] = (
    "patient_ops.db",
    "patient_ops.adapters",
    # The booking desk: holds, the calendar write, the read-back. tools/ is
    # otherwise where agents are *meant* to reach data, so this one module is
    # named on its own -- the read-only toolset stays importable.
    "patient_ops.tools.booking",
    # The booking nodes, which hold the desk. An agent that imported the graph
    # could call execute_booking directly.
    "patient_ops.graph",
    # Retrieval is read-only, but rag/ingest.py writes, and an agent that can
    # reach the corpus can rewrite what it is later grounded against. Agents
    # search through the read-only tools in tools/ instead.
    "patient_ops.rag",
    "patient_ops.redis_layer",
    "patient_ops.jobs",
    "sqlalchemy",
    "psycopg",
    "psycopg_pool",
    "redis",
    "arq",
)

# Two pattern groups, because the two kinds of write hide in different places.
#
# CODE patterns are method and function calls. They are matched against source
# with comments and string bodies stripped, so a docstring that *describes* a
# write ("agents never call book_appointment()") does not trip the check.
#
# SQL patterns are the opposite case: raw SQL only ever appears *inside* a
# string literal, so stripping strings would hide every one of them. These are
# matched against string contents instead.
CODE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.\s*(commit|flush)\s*\("), "database transaction commit"),
    (re.compile(r"\bsession\s*\.\s*(add|add_all|merge|delete)\s*\("), "ORM write"),
    (re.compile(r"\.\s*(set|setex|setnx|getset|mset|incr|decr|expire)\s*\(", re.I), "Redis write"),
    (re.compile(r"\.\s*(hset|hdel|lpush|rpush|sadd|srem|zadd|zrem|unlink)\s*\("), "Redis write"),
    (re.compile(r"\benqueue_job\s*\("), "queue write (the non-critical path is still a write)"),
    (re.compile(r"\b(book_appointment|place_holds?|release_holds)\s*\("), "booking write"),
    (re.compile(r"\bescalate_to_human\s*\("), "escalation write -- agents emit Handoff instead"),
    (re.compile(r"\.\s*(post|put|patch|delete)\s*\("), "outbound mutating HTTP call"),
    (re.compile(r"\b(import_module|__import__)\s*\("), "dynamic import -- defeats the import rule"),
]

SQL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bINSERT\s+INTO\b", re.I), "raw SQL INSERT"),
    (re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.I), "raw SQL UPDATE"),
    (re.compile(r"\bDELETE\s+FROM\b", re.I), "raw SQL DELETE"),
    (re.compile(r"\b(TRUNCATE|DROP)\s+TABLE\b", re.I), "raw SQL DDL"),
    (re.compile(r"\bSELECT\b.*\bFOR\s+UPDATE\b", re.I | re.S), "row lock (a write intent)"),
]


@dataclass(frozen=True)
class Violation:
    path: Path
    line_no: int
    rule: str
    source: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_relative_to(REPO_ROOT) else self.path
        return f"  {rel}:{self.line_no}\n      {self.rule}: {self.source.strip()}"


def package_of(path: Path) -> str:
    """Dotted package a source file belongs to, e.g. patient_ops.agents."""
    return ".".join(path.relative_to(SRC_ROOT).parent.parts)


def _imported_modules(node: ast.Import | ast.ImportFrom, package: str) -> list[str]:
    """Every module an import statement can bind, with relative imports resolved.

    `from patient_ops import db` binds the module patient_ops.db, so each
    imported name is also checked as a submodule.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:  # relative: one leading dot = this package, two = its parent, ...
        parts = package.split(".")
        anchor = ".".join(parts[: len(parts) - node.level + 1])
        base = f"{anchor}.{base}" if base else anchor
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def _denied_prefix(module: str) -> str | None:
    return next((d for d in DENIED_IMPORTS if module == d or module.startswith(d + ".")), None)


def _dotted_name(node: ast.Attribute) -> str | None:
    """`patient_ops.db.repo.find` -> "patient_ops.db.repo.find"; None if not a plain chain."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return ".".join([current.id, *reversed(parts)])


def scan_imports(path: Path, package: str) -> list[Violation]:
    """Imports of write-capable modules, and references that reach them anyway.

    The second case closes a bypass: `import patient_ops` imports nothing
    denied, yet `patient_ops.db.repo...` then walks straight into the database
    layer. Only our own package is checked this way -- a third-party module
    cannot be reached without importing it, which the first case catches.
    """
    source = path.read_text(encoding="utf-8")
    raw_lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the pattern scan still covers the raw text

    found: dict[int, str] = {}  # line -> rule; one report per line
    # walk, not body: TYPE_CHECKING blocks and function-local imports count too
    for node in ast.walk(tree):
        rule = None
        if isinstance(node, ast.Import | ast.ImportFrom):
            denied = next(filter(None, map(_denied_prefix, _imported_modules(node, package))), None)
            rule = f"import of write-capable {denied}" if denied else None
        elif isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            denied = (
                _denied_prefix(dotted) if dotted and dotted.startswith("patient_ops.") else None
            )
            rule = f"reference to write-capable {denied}" if denied else None
        if rule and node.lineno not in found and ALLOW_MARKER not in raw_lines[node.lineno - 1]:
            found[node.lineno] = rule

    return [Violation(path, n, rule, raw_lines[n - 1]) for n, rule in sorted(found.items())]


def tokenize_source(source: str) -> tuple[dict[int, str], dict[int, str]]:
    """Split a module into (code-only text, string-literal text), keyed by line.

    Returning both is what lets the two pattern groups be checked against the
    right half of the file -- see CODE_PATTERNS / SQL_PATTERNS.
    """
    code: dict[int, str] = {}
    strings: dict[int, str] = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable file: check the raw text rather than skipping it silently.
        raw = dict(enumerate(source.splitlines(), start=1))
        return raw, raw

    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    for tok in tokens:
        if tok.type in skip:
            continue
        line_no = tok.start[0]
        if tok.type == tokenize.STRING:
            strings[line_no] = strings.get(line_no, "") + tok.string + " "
            code[line_no] = code.get(line_no, "") + '"" '
        else:
            code[line_no] = code.get(line_no, "") + tok.string + " "
    return code, strings


def scan_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    raw_lines = source.splitlines()
    code_lines, string_lines = tokenize_source(source)

    violations: list[Violation] = []
    for line_no in sorted(set(code_lines) | set(string_lines)):
        raw = raw_lines[line_no - 1] if line_no <= len(raw_lines) else ""
        if ALLOW_MARKER in raw:
            continue
        checks = (
            (CODE_PATTERNS, code_lines.get(line_no, "")),
            (SQL_PATTERNS, string_lines.get(line_no, "")),
        )
        hit = next(
            (
                rule
                for patterns, text in checks
                if text
                for pattern, rule in patterns
                if pattern.search(text)
            ),
            None,
        )
        if hit:
            violations.append(Violation(path, line_no, hit, raw))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not GUARDED_DIR.is_dir():
        print(f"✗ guarded directory missing: {GUARDED_DIR}", file=sys.stderr)
        return 1

    files = sorted(p for p in GUARDED_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    violations = [v for f in files for v in (*scan_imports(f, package_of(f)), *scan_file(f))]

    rel = GUARDED_DIR.relative_to(REPO_ROOT)
    if args.verbose:
        for f in files:
            print(f"  scanned {f.relative_to(REPO_ROOT)}")

    if violations:
        print(f"✗ {len(violations)} write operation(s) found in {rel}/\n", file=sys.stderr)
        for v in violations:
            print(v.render(), file=sys.stderr)
        print(
            "\n  Agents are read-only by design. Reach data through the read-only tools in\n"
            "  tools/, move writes behind the deterministic validate -> execute -> verify\n"
            "  path in graph/nodes/, or mark a reviewed exception with\n"
            f"  `# {ALLOW_MARKER} -- <reason>`.",
            file=sys.stderr,
        )
        return 1

    print(f"✓ no write operations in {rel}/ ({len(files)} file(s) scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
