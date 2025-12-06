#!/bin/bash
# Run SwiftLint auto-fixes safely
#
# Usage:
#   ./Scripts/lint_autofix.sh [--all]
#
# Without --all, only runs safe auto-corrections.
# With --all, runs all auto-corrections (review diff carefully!)

set -e

cd "$(dirname "$0")/.."

echo "=== SwiftLint Auto-Fix ==="
echo ""

# Check swiftlint is installed
if ! command -v swiftlint &> /dev/null; then
    echo "Error: swiftlint not installed. Run: brew install swiftlint"
    exit 1
fi

# Show current state
echo "Current violations:"
swiftlint lint --quiet 2>&1 | tail -5
echo ""

# Safe auto-corrections (won't break code)
SAFE_RULES=(
    "trailing_newline"
    "trailing_comma"
    "vertical_whitespace"
    "vertical_whitespace_closing_braces"
    "redundant_discardable_let"
    "redundant_void_return"
    "empty_enum_arguments"
    "unneeded_parentheses_in_closure_argument"
    "modifier_order"
    "operator_usage_whitespace"
)

if [[ "$1" == "--all" ]]; then
    echo "Running ALL auto-corrections..."
    echo "WARNING: Review the diff carefully before committing!"
    echo ""
    swiftlint --fix --format
else
    echo "Running safe auto-corrections only..."
    echo "(Use --all for aggressive fixes)"
    echo ""

    # SwiftLint doesn't support per-rule fixing, so we use --fix
    # but the config already limits which rules can auto-correct
    swiftlint --fix
fi

echo ""
echo "=== After auto-fix ==="
swiftlint lint --quiet 2>&1 | tail -5

echo ""
echo "Review changes with: git diff"
echo "Run tests with: swift test --parallel"
