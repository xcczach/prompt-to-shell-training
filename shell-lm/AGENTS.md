# shell-lm Agent Notes

- Scope: All files within `shell-lm/`.
- Follow project spec in repository root AGENTS.md. If conflicts arise, direct user instructions take precedence.

Conventions
- Python: type hints, argparse CLI, docstrings, stdout logging, non-zero exit on failure.
- Keep external dependencies optional with graceful fallbacks and clear error messages.
- Prefer safe/read-only operations when synthesizing commands; set `requires_confirm=true` for potentially destructive operations.
- PowerShell usage must load profiles; do NOT pass `-NoProfile` in any script call. Use `pwsh` when available, fallback to `powershell`.
- Shellcheck is optional; if missing, skip with a warning. Same for Docker.

Data & Ratios
- Language: 50% Chinese / 50% English.
- OS: Linux 45%, macOS 25%, Windows 30%.
- Shell mix per OS: Linux (bash 70%, zsh 20%, fish 10%), macOS (zsh 80%, bash 20%), Windows (PowerShell 80%, cmd 20%).

Testing
- Scripts should work cross-platform and avoid hard-coding paths.
- Do not attempt to execute user commands outside of Docker in `exec_sandbox.py`.

