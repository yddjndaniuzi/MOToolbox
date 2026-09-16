#!/usr/bin/env python3
"""Route pushes according to a conservative sensitive-content scan."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ZERO_SHA = "0" * 40
PUBLIC_REMOTE = "origin"
PRIVATE_REMOTE = "private"
REVIEW_RULE_EXEMPT_PATHS = {
    "scripts/safe_push.py",
    "tests/test_safe_push.py",
}

SENSITIVE_PATH_PATTERNS = (
    re.compile(r"(^|/)\.env(?:\.|$)", re.IGNORECASE),
    re.compile(r"(^|/)pressconf/config(?:/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(?:private|internal|confidential)(?:/|$)", re.IGNORECASE),
    re.compile(r"\.(?:pem|key|p12|pfx|keystore|jks|docx|xlsx|xlsm|pptx|pdf|csv|tsv|mp3|wav|m4a|mp4|mov)$", re.IGNORECASE),
)

HARD_SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b")),
    ("cloud-access-key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("model-api-key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b")),
)

GENERIC_SENSITIVE_PATTERNS = (
    ("confidential-marker", re.compile(r"仅供内部|不得外传|严禁外传|内部资料|内部文档|保密|机密|未发布|under\s+nda|confidential", re.IGNORECASE)),
    ("internal-host", re.compile(r"https?://[^\s/]*(?:\.internal|\.corp)(?:[/:]|$)", re.IGNORECASE)),
    ("company-email", re.compile(r"[A-Za-z0-9._%+-]+@(?:[^\s@]+\.)?(?:internal|corp)\b", re.IGNORECASE)),
)

ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|app[_-]?secret|client[_-]?secret|access[_-]?token|refresh[_-]?token|password|passwd|authorization)"
    r"\s*[=:]\s*['\"]([^'\"]{12,})['\"]"
)
PLACEHOLDER_PATTERN = re.compile(
    r"example|sample|placeholder|redacted|dummy|test|your[_-]|<|\{\{|os\.environ|getenv|form\.get|config",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    location: str


def run_git(*args: str, check: bool = True, capture: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def repo_root() -> Path:
    return Path(run_git("rev-parse", "--show-toplevel").stdout.strip())


def current_branch() -> str:
    return run_git("branch", "--show-current").stdout.strip()


def configured_local_patterns(root: Path) -> list[tuple[str, re.Pattern[str]]]:
    result = run_git("config", "--get", "motoolbox.sensitivePatternsFile", check=False)
    configured = result.stdout.strip()
    if not configured:
        return []
    path = Path(configured)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return []
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append((f"local-pattern-{index}", re.compile(line, re.IGNORECASE)))
        except re.error as exc:
            raise RuntimeError(f"敏感规则文件第 {index} 行不是有效正则：{exc}") from exc
    return patterns


def changed_files(base: str, tip: str) -> list[str]:
    result = run_git("diff", "--name-only", "--diff-filter=ACMRT", f"{base}..{tip}", "--")
    return [line for line in result.stdout.splitlines() if line]


def added_lines(base: str, tip: str) -> list[tuple[str, str]]:
    result = run_git("diff", "--no-ext-diff", "--unified=0", "--diff-filter=ACMRT", f"{base}..{tip}", "--")
    current_file = "<unknown>"
    additions: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if line.startswith("+") and not line.startswith("+++"):
            additions.append((current_file, line[1:]))
    return additions


def scan_text(location: str, text: str, local_patterns: list[tuple[str, re.Pattern[str]]] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for rule, pattern in HARD_SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(Finding("hard", rule, location))
    for match in ASSIGNMENT_PATTERN.finditer(text):
        if not PLACEHOLDER_PATTERN.search(match.group(1)):
            findings.append(Finding("hard", "credential-assignment", location))
            break
    for rule, pattern in GENERIC_SENSITIVE_PATTERNS:
        if pattern.search(text):
            findings.append(Finding("review", rule, location))
    for rule, pattern in local_patterns or []:
        if pattern.search(text):
            findings.append(Finding("review", rule, location))
    return findings


def scan_range(base: str, tip: str, root: Path | None = None) -> list[Finding]:
    root = root or repo_root()
    findings: list[Finding] = []
    for path in changed_files(base, tip):
        if any(pattern.search(path) for pattern in SENSITIVE_PATH_PATTERNS):
            findings.append(Finding("review", "sensitive-path", path))
    local_patterns = configured_local_patterns(root)
    for path, line in added_lines(base, tip):
        line_findings = scan_text(path, line, local_patterns)
        if path in REVIEW_RULE_EXEMPT_PATHS:
            line_findings = [finding for finding in line_findings if finding.severity == "hard"]
        findings.extend(line_findings)
    messages = run_git("log", "--format=%B", f"{base}..{tip}").stdout
    findings.extend(scan_text("commit-message", messages, local_patterns))
    return sorted(set(findings), key=lambda item: (item.severity, item.rule, item.location))


def print_findings(findings: list[Finding]) -> None:
    print("检测到可能不适合公开的新增内容：", file=sys.stderr)
    for finding in findings:
        label = "密钥/凭据" if finding.severity == "hard" else "需人工复核"
        print(f"  - [{label}] {finding.rule}: {finding.location}", file=sys.stderr)


def push_private(refspecs: list[str]) -> None:
    run_git("push", PRIVATE_REMOTE, *refspecs, capture=False)


def dispatch(allow_review: bool = False, force_public: bool = False) -> int:
    branch = current_branch()
    if not branch:
        print("当前为 detached HEAD，默认只允许显式推送到私密仓库。", file=sys.stderr)
        return 2
    local_ref = f"refs/heads/{branch}"
    refspec = f"{local_ref}:{local_ref}"
    if branch != "main":
        push_private([refspec])
        print(f"分支 {branch} 默认只推送到私密仓库。")
        return 0

    run_git("fetch", "--quiet", PUBLIC_REMOTE, "main", capture=False)
    base = run_git("rev-parse", f"{PUBLIC_REMOTE}/main").stdout.strip()
    tip = run_git("rev-parse", "HEAD").stdout.strip()
    findings = scan_range(base, tip)
    hard = [finding for finding in findings if finding.severity == "hard"]
    blocked = bool(hard and not force_public) or bool(findings and not (allow_review or force_public))
    if blocked:
        print_findings(findings)
        push_private([refspec])
        print("已只推送到私密仓库；公开仓库未更新。")
        return 0

    env = os.environ.copy()
    env["MOTOOLBOX_SAFE_PUSH_ACTIVE"] = "1"
    push_private([refspec])
    run_git("push", PUBLIC_REMOTE, refspec, capture=False, env=env)
    if findings:
        print("已按人工覆盖选项同步到公开与私密仓库。")
    else:
        print("扫描通过，已同步到公开与私密仓库。")
    return 0


def hook(remote_name: str, updates: list[str]) -> int:
    if remote_name != PUBLIC_REMOTE or os.environ.get("MOTOOLBOX_SAFE_PUSH_ACTIVE") == "1":
        return 0
    root = repo_root()
    refspecs: list[str] = []
    findings: list[Finding] = []
    for update in updates:
        parts = update.split()
        if len(parts) != 4:
            continue
        local_ref, local_sha, remote_ref, remote_sha = parts
        if local_sha == ZERO_SHA:
            findings.append(Finding("review", "public-ref-deletion", remote_ref))
            continue
        if remote_sha == ZERO_SHA:
            findings.append(Finding("review", "new-public-ref", remote_ref))
        else:
            findings.extend(scan_range(remote_sha, local_sha, root))
        refspecs.append(f"{local_ref}:{remote_ref}")

    if refspecs:
        try:
            push_private(refspecs)
        except subprocess.CalledProcessError:
            print("私密仓库同步失败，已取消公开推送。", file=sys.stderr)
            return 1
    if findings:
        print_findings(sorted(set(findings), key=lambda item: (item.severity, item.rule, item.location)))
        print("内容已同步到私密仓库；公开推送已由安全钩子取消。", file=sys.stderr)
        print("确认属于误报时，请使用 git safe-push --public；密钥类误报需使用 --force-public。", file=sys.stderr)
        return 1
    print("安全扫描通过；私密仓库已先行同步，继续公开推送。", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="人工复核后允许普通敏感词命中公开推送")
    parser.add_argument("--force-public", action="store_true", help="明确覆盖包括凭据规则在内的全部阻断")
    parser.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("hook_remote", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("hook_url", nargs="?", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hook:
        return hook(args.hook_remote or "", sys.stdin.read().splitlines())
    return dispatch(allow_review=args.public, force_public=args.force_public)


if __name__ == "__main__":
    raise SystemExit(main())
