"""Common utilities for the shell-lm pipeline.

Includes:
- JSONL helpers
- Command canonicalization
- Safety pattern checks
- Shell/OS constants
- PowerShell detection (pwsh vs powershell)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ALLOWED_TYPES = {"bash", "zsh", "fish", "powershell", "cmd", "nushell"}

OS_RATIOS = {"linux": 45, "macos": 25, "windows": 30}
SHELL_MIX = {
    "linux": {"bash": 70, "zsh": 20, "fish": 10},
    "macos": {"zsh": 80, "bash": 20},
    "windows": {"powershell": 80, "cmd": 20},
}


def jsonl_read(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def jsonl_write(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


RISKY_PATTERNS = [
    r"\brm\s+-rf\s+/(\s|$)",
    r"\bdd\b.*\b(of=|if=)/dev/(sd|nvme|hd)",
    r"\bmkfs(\.|\b)",
    r":\(\)\{\s*:\|:&\s*\};:",
    r"\bshutdown\b",
    r"\b(secretsdump|mimikatz|hashdump|laZagne)\b",
    r"\bnmap\b.*\s-\w*(A|sS|sU)\b",
]


def looks_risky(cmd: str) -> bool:
    s = cmd.strip().lower()
    for pat in RISKY_PATTERNS:
        if re.search(pat, s):
            return True
    return False


_MULTISPACE_RE = re.compile(r"\s+")


def canonicalize_cmd(cmd: str) -> str:
    """Basic canonicalization for deduplication.
    - Strip outer whitespace
    - Collapse multiple spaces
    - Remove redundant trailing semicolons
    - Normalize quotes spacing
    Note: conservative to avoid changing semantics.
    """
    s = cmd.strip().replace("\r", "")
    s = s.rstrip("; ")
    # Normalize whitespace around pipes and redirects
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*&&\s*", " && ", s)
    s = re.sub(r"\s*\|\|\s*", " || ", s)
    # Collapse spaces
    s = _MULTISPACE_RE.sub(" ", s)
    return s


def which_powershell() -> Optional[str]:
    # Prefer pwsh if available; else fallback to powershell (Windows PowerShell)
    for exe in ("pwsh", "powershell"):
        path = shutil.which(exe)
        if path:
            return path
    return None


def ensure_allowed_type(t: str) -> bool:
    return t in ALLOWED_TYPES


def validate_output_json(obj: dict) -> Tuple[bool, Optional[str]]:
    for k in ("type", "cmd", "requires_confirm", "explain"):
        if k not in obj:
            return False, f"missing field: {k}"
    if not ensure_allowed_type(obj["type"]):
        return False, f"invalid type: {obj['type']}"
    if not isinstance(obj["requires_confirm"], bool):
        return False, "requires_confirm must be boolean"
    if not isinstance(obj["cmd"], str) or not obj["cmd"].strip():
        return False, "cmd must be non-empty string"
    if not isinstance(obj["explain"], str) or not obj["explain"].strip():
        return False, "explain must be non-empty string"
    return True, None


def family_from_cmd(cmd: str, shell_type: str) -> str:
    s = canonicalize_cmd(cmd).strip()
    if not s:
        return "other"
    # First token heuristic (handles PowerShell verbs and UNIX tools)
    first = re.split(r"\s|;|\||&&|\|\|", s, maxsplit=1)[0].strip().lower()
    if shell_type == "powershell" and re.match(r"^[a-z]+-[a-z]+$", first):
        return first.split("-")[0]  # verb as family
    # Map common tools to families
    maps = {
        "git": "git", "grep": "grep", "find": "find", "sed": "sed", "awk": "awk",
        "curl": "curl", "wget": "wget", "ls": "ls", "dir": "dir", "tree": "tree",
        "brew": "brew", "apt": "apt", "dnf": "dnf", "yum": "yum", "pacman": "pacman",
        "winget": "winget", "choco": "choco", "scoop": "scoop", "pip": "pip",
        "python": "python", "node": "node", "npm": "npm", "pnpm": "pnpm", "yarn": "yarn",
        "docker": "docker", "kubectl": "kubectl", "helm": "helm", "gh": "gh",
    }
    return maps.get(first, first or "other")


def stratified_quota_counts(total: int, os_ratios: Dict[str, int] = OS_RATIOS, shell_mix: Dict[str, Dict[str, int]] = SHELL_MIX) -> Dict[Tuple[str, str], int]:
    # Compute per (os,shell) counts rounding to sum total
    per_os = {osk: round(total * pct / 100) for osk, pct in os_ratios.items()}
    # Fix rounding to exact total
    delta = total - sum(per_os.values())
    if delta != 0:
        # Assign remainder to largest buckets deterministically
        keys_sorted = sorted(per_os.keys(), key=lambda k: (-os_ratios[k], k))
        for i in range(abs(delta)):
            idx = i % len(keys_sorted)
            per_os[keys_sorted[idx]] += 1 if delta > 0 else -1
    per_bucket: Dict[Tuple[str, str], int] = {}
    for osk, os_cnt in per_os.items():
        mix = shell_mix[osk]
        per_shell = {sh: round(os_cnt * pct / 100) for sh, pct in mix.items()}
        d = os_cnt - sum(per_shell.values())
        if d != 0:
            ks = sorted(per_shell.keys(), key=lambda k: (-mix[k], k))
            for i in range(abs(d)):
                idx = i % len(ks)
                per_shell[ks[idx]] += 1 if d > 0 else -1
        for sh, c in per_shell.items():
            per_bucket[(osk, sh)] = c
    return per_bucket


