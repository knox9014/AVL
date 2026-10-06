"""Inspect Git index text without printing matched private values.

This is a publication aid, not proof that arbitrary data contains no PII.
No network access or model dependency is required.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


PATTERNS = {
    'local_user_path': re.compile(r'(?:[A-Za-z]:[\\/]Users[\\/][^\s\\/]+|/(?:home|Users)/[^\s/]+)', re.I),
    'email_address': re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])'),
    'api_token': re.compile(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
}
NEUTRAL_EMAIL = 'avl@users.noreply.github.com'
RESERVED_DOMAINS = {'example.com', 'example.org', 'example.net'}


def local_identities():
    """Read current OS identifiers only; never save them into a report."""
    values = [os.environ.get('USERNAME'), os.environ.get('USER')]
    # Generic service accounts are not personal identifiers.
    return tuple(sorted({v for v in values if v and len(v) >= 3 and
                         v.casefold() not in {'root', 'system', 'admin', 'administrator', 'runner'}}))


def _categories(text, identities):
    found = set()
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if category == 'email_address':
                address = match.group().casefold()
                if address == NEUTRAL_EMAIL or address.rsplit('@', 1)[1] in RESERVED_DOMAINS:
                    continue
            found.add(category)
            break
    for identity in identities:
        if re.search(r'(?<![\w])' + re.escape(identity) + r'(?![\w])', text, re.I):
            found.add('local_identity')
    return sorted(found)


def scan_bytes(filename, data, *, identities=None):
    """Return findings and optional binary advisory; source text is never returned."""
    identities = local_identities() if identities is None else identities
    path_categories = _categories(filename, identities)
    label = '<redacted file>' if path_categories else filename
    findings = [{'file': label, 'line': 0, 'category': category} for category in path_categories]
    try:
        if b'\x00' in data:
            raise UnicodeError()
        text = data.decode('utf-8-sig')
    except UnicodeError:
        return findings, {'file': label, 'bytes': len(data), 'reason': 'binary_not_inspected'}
    for number, line in enumerate(text.splitlines(), 1):
        findings.extend({'file': label, 'line': number, 'category': category}
                        for category in _categories(line, identities))
    return findings, None


def _git(root, *args):
    process = subprocess.run(['git', *args], cwd=root, capture_output=True)
    if process.returncode:
        # Git stderr may contain local paths. Do not echo it into public logs.
        raise RuntimeError('Git inspection failed; no publication decision was made.')
    return process.stdout


def inspect_git(root, scope='staged', *, identities=None):
    """Inspect staged additions/changes or all tracked index blobs, never working files."""
    if scope not in {'staged', 'tracked'}:
        raise ValueError('scope must be staged or tracked')
    names = (_git(root, 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z')
             if scope == 'staged' else _git(root, 'ls-files', '-z'))
    filenames = sorted(set(name.decode('utf-8', errors='surrogateescape')
                           for name in names.split(b'\x00') if name))
    findings, skipped = [], []
    for filename in filenames:
        data = _git(root, 'show', ':' + filename)
        issues, advisory = scan_bytes(filename, data, identities=identities)
        findings.extend(issues)
        if advisory:
            skipped.append(advisory)
    return {'scope': scope, 'files_checked': len(filenames), 'findings': findings,
            'binary_skips': skipped, 'blocked': bool(findings),
            'limitation': 'Text pattern scan only; binary skips and semantic PII require separate review.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--staged', action='store_true', help='Scan staged changes (default).')
    group.add_argument('--tracked', action='store_true', help='Scan all tracked index blobs.')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = inspect_git(args.root, 'tracked' if args.tracked else 'staged')
    except (RuntimeError, OSError):
        print(json.dumps({'blocked': True, 'error': 'Git inspection unavailable; publication must wait.'}))
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return int(report['blocked'])


if __name__ == '__main__':
    raise SystemExit(main())
