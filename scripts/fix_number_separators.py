#!/usr/bin/env python3
"""
Fix SwiftLint number_separator violations by adding underscores to large numbers.

Usage:
    python3 Scripts/fix_number_separators.py [--dry-run]

Examples:
    1000000 -> 1_000_000
    0.000001 -> 0.000_001
    0x1000000 -> 0x100_0000
"""

import re
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

def add_separators(match: re.Match) -> str:
    """Add underscores to a number literal."""
    num = match.group(0)

    # Skip if already has separators
    if "_" in num:
        return num

    # Handle hex numbers
    if num.startswith("0x") or num.startswith("0X"):
        hex_part = num[2:]
        if len(hex_part) >= 5:
            # Group hex digits in fours from right
            separated = "_".join(
                hex_part[max(0, i-4):i]
                for i in range(len(hex_part), 0, -4)
            )[::-1].replace("_", "", 1)[::-1]
            # Simpler approach: insert every 4 chars from right
            result = ""
            for i, c in enumerate(reversed(hex_part)):
                if i > 0 and i % 4 == 0:
                    result = "_" + result
                result = c + result
            return num[:2] + result
        return num

    # Handle floating point
    if "." in num or "e" in num.lower():
        return num  # Skip floats for safety

    # Handle integers - only if 5+ digits (per swiftlint config)
    if len(num) < 5:
        return num

    # Add separators every 3 digits from right
    result = ""
    for i, c in enumerate(reversed(num)):
        if i > 0 and i % 3 == 0:
            result = "_" + result
        result = c + result

    return result

def process_file(path: Path) -> tuple[int, list[str]]:
    """Process a single Swift file. Returns (changes_count, change_descriptions)."""
    content = path.read_text()
    changes = []

    # Pattern for integer literals with 5+ digits
    # Negative lookbehind for decimal point to avoid matching fractional parts
    # Negative lookahead for decimal point to avoid matching before fractions
    pattern = r'(?<![.\d])(\d{5,})(?![.\d])'

    def replace_with_tracking(match):
        original = match.group(0)
        replaced = add_separators(match)
        if original != replaced:
            changes.append(f"  {original} -> {replaced}")
        return replaced

    new_content = re.sub(pattern, replace_with_tracking, content)

    if changes and not DRY_RUN:
        path.write_text(new_content)

    return len(changes), changes

def main():
    sources = Path("Sources")
    tests = Path("Tests")

    if not sources.exists():
        print("Error: Run from repository root")
        sys.exit(1)

    total_changes = 0
    files_changed = 0

    for folder in [sources, tests]:
        for swift_file in folder.rglob("*.swift"):
            count, changes = process_file(swift_file)
            if count > 0:
                files_changed += 1
                total_changes += count
                print(f"{swift_file}: {count} changes")
                if DRY_RUN:
                    for change in changes[:5]:
                        print(change)
                    if len(changes) > 5:
                        print(f"  ... and {len(changes) - 5} more")

    print(f"\n{'Would fix' if DRY_RUN else 'Fixed'} {total_changes} numbers in {files_changed} files")
    if DRY_RUN:
        print("Run without --dry-run to apply changes")

if __name__ == "__main__":
    main()
