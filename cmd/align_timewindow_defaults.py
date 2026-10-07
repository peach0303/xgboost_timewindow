#!/usr/bin/env python3
"""Align scalar csh defaults in timewindow_common.csh with the main pipeline.

The main pipeline is the source of truth.  By default this program only
reports differences.  Pass --apply to update defaults which are declared in
both files; variables unique to either file are intentionally not added or
removed.

Examples:
    python cmd/align_timewindow_defaults.py
    python cmd/align_timewindow_defaults.py --apply
"""

import argparse
import re
import sys
from pathlib import Path


# Matches the one-line csh convention used by both configuration files:
# if ( ! $?NAME ) set NAME = value
DEFAULT_RE = re.compile(
    r"^(?P<prefix>\s*if\s*\(\s*!\s*\$\?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
    r"\s*set\s+(?P=name)\s*=\s*)(?P<value>.*?)(?P<newline>\r?\n?)$"
)
ASSIGNMENT_RE = re.compile(
    r"^\s*set\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)(?:\r?\n)?$"
)
VARIABLE_RE = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")


def read_defaults(path: Path):
    """Return lines, defaults, and simple assignments from a csh file."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    defaults = {}
    assignments = {}
    for index, line in enumerate(lines):
        match = DEFAULT_RE.match(line)
        if match:
            defaults[match.group("name")] = (index, match.group("value"))
        assignment = ASSIGNMENT_RE.match(line)
        if assignment:
            assignments[assignment.group("name")] = assignment.group("value")
    return lines, defaults, assignments


def resolve_variables(value: str, assignments: dict, seen=frozenset()) -> str:
    """Resolve simple $NAME/${NAME} aliases without executing csh code."""
    def replace(match):
        name = match.group("braced") or match.group("plain")
        if name in seen or name not in assignments:
            return match.group(0)
        return resolve_variables(assignments[name], assignments, seen | {name})

    return VARIABLE_RE.sub(replace, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path(__file__).with_name("run_timewindow_pipeline.csh"),
        help="source-of-truth pipeline file",
    )
    parser.add_argument(
        "--common",
        type=Path,
        default=Path(__file__).with_name("timewindow_common.csh"),
        help="common defaults file to compare/update",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write pipeline values into same-named common defaults",
    )
    args = parser.parse_args()

    for path in (args.pipeline, args.common):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    pipeline_lines, pipeline, pipeline_assignments = read_defaults(args.pipeline)
    common_lines, common, common_assignments = read_defaults(args.common)
    shared = sorted(set(pipeline) & set(common))
    assignments = dict(common_assignments)
    assignments.update(pipeline_assignments)
    differences = []
    for name in shared:
        pipeline_value = resolve_variables(pipeline[name][1], assignments).replace(chr(34), "")
        common_value = resolve_variables(common[name][1], assignments).replace(chr(34), "")
        if pipeline_value != common_value:
            differences.append(name)

    print(f"pipeline: {args.pipeline}")
    print(f"common:   {args.common}")
    print(f"shared scalar defaults: {len(shared)}")
    if differences:
        print(f"different defaults: {len(differences)}")
        for name in differences:
            print(f"  {name}: common={common[name][1]!r} -> pipeline={pipeline[name][1]!r}")
    else:
        print("different defaults: 0 (already aligned)")

    pipeline_only = sorted(set(pipeline) - set(common))
    common_only = sorted(set(common) - set(pipeline))
    if pipeline_only:
        print("pipeline-only defaults (not added): " + ", ".join(pipeline_only))
    if common_only:
        print("common-only defaults (kept): " + ", ".join(common_only))

    if not args.apply:
        return 1 if differences else 0

    common_symbols = set(common) | set(common_assignments)
    apply_names = []
    skipped_names = []
    for name in differences:
        references = {match.group("braced") or match.group("plain") for match in VARIABLE_RE.finditer(pipeline[name][1])}
        if references - common_symbols:
            skipped_names.append(name)
        else:
            apply_names.append(name)
    for name in apply_names:
        index, _ = common[name]
        match = DEFAULT_RE.match(common_lines[index])
        assert match is not None
        common_lines[index] = match.group("prefix") + pipeline[name][1] + match.group("newline")
    if apply_names:
        args.common.write_text("".join(common_lines), encoding="utf-8")
    if skipped_names:
        print("not updated (pipeline refers to common-undefined variable): " + ", ".join(skipped_names))
    print(f"updated {len(apply_names)} default(s) in {args.common}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
