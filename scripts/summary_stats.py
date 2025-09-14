#!/usr/bin/env python3
"""
Compute summary statistics for 'completion' fields in evaluation JSONL outputs.

Usage:
    python scripts/summary_stats.py --input /path/to/eval.jsonl [--top_k 20] [--lowercase] [--strip] [--json]

Reads a JSONL where each line is an object with key 'responses', which is a list
of entries shaped like {"response": {"completion": "...", ...}, ...} and
aggregates statistics over the completion strings.
"""

import argparse
import json
from collections import Counter
from typing import Iterable, List, Optional


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                continue


def extract_completions(obj: dict) -> List[str]:
    completions: List[str] = []
    responses = obj.get("responses")
    if not isinstance(responses, list):
        return completions
    for entry in responses:
        if not isinstance(entry, dict):
            continue
        response = entry.get("response")
        if not isinstance(response, dict):
            continue
        completion = response.get("completion")
        if isinstance(completion, str):
            completions.append(completion)
    return completions


def normalize_text(text: str, do_lower: bool, do_strip: bool) -> str:
    if do_strip:
        text = text.strip()
    if do_lower:
        text = text.lower()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 'completion' field statistics from evaluation JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K most frequent completions to show")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase completions before counting")
    parser.add_argument("--strip", action="store_true", help="Strip whitespace around completions before counting")
    parser.add_argument("--json", action="store_true", help="Output stats as JSON instead of text")

    args = parser.parse_args()

    all_completions: List[str] = []
    for obj in iter_jsonl(args.input):
        all_completions.extend(extract_completions(obj))

    normalized: List[str] = [
        normalize_text(c, do_lower=args.lowercase, do_strip=args.strip) for c in all_completions
    ]

    # Filter out empty strings after normalization
    normalized = [c for c in normalized if c != ""]

    count = len(normalized)
    unique_count = len(set(normalized))
    lengths = [len(c) for c in normalized]
    avg_len: Optional[float] = (sum(lengths) / count) if count > 0 else None
    min_len: Optional[int] = min(lengths) if lengths else None
    max_len: Optional[int] = max(lengths) if lengths else None

    freq = Counter(normalized)
    top_k = freq.most_common(args.top_k)

    if args.json:
        output = {
            "count": count,
            "unique_count": unique_count,
            "avg_length": avg_len,
            "min_length": min_len,
            "max_length": max_len,
            "top_k": top_k,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Human-readable output
    print(f"Total completions: {count}")
    print(f"Unique completions: {unique_count}")
    print(f"Average length: {avg_len}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print("")
    print(f"Top {args.top_k} completions:")
    for value, ct in top_k:
        # Show a safe preview in case of long values
        preview = value if len(value) <= 80 else (value[:77] + "...")
        print(f"{ct:>7}  | {preview}")


if __name__ == "__main__":
    main()
