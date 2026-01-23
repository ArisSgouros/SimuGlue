from __future__ import annotations

import re
from typing import Sequence


def _format_cell_block(cell: Sequence[Sequence[float]]) -> list[str]:
    if len(cell) != 3 or any(len(row) != 3 for row in cell):
        raise ValueError("cell must be a 3x3 array-like.")
    lines = ["CELL_PARAMETERS {angstrom}"]
    for vec in cell:
        lines.append(f"{vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}")
    return lines


def _format_positions_block(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
) -> list[str]:
    if len(symbols) != len(positions):
        raise ValueError("Mismatch between number of symbols and positions.")
    lines = ["ATOMIC_POSITIONS {angstrom}"]
    for s, (x, y, z) in zip(symbols, positions):
        # trailing space kept (matches your previous style)
        lines.append(f"{s} {x:.16f} {y:.16f} {z:.16f} ")
    return lines


def _find_card_block(
    lines: list[str],
    header_pattern: re.Pattern,
    n_body_lines_hint: int | None = None,
) -> tuple[int | None, int | None]:
    """
    Find [start, end) indices of a QE card-like block.

    If n_body_lines_hint is given:
      - block = header line + next `n_body_lines_hint` non-empty lines.
    Otherwise:
      - block = header line + subsequent non-empty, non-new-card lines.
    """
    start = None
    for i, line in enumerate(lines):
        if header_pattern.match(line):
            start = i
            break

    if start is None:
        return None, None

    # If we know how many lines to consume (CELL: 3, ATOMIC_POSITIONS: nat)
    if n_body_lines_hint is not None:
        j = start + 1
        count = 0
        while j < len(lines) and count < n_body_lines_hint:
            if lines[j].strip() != "":
                count += 1
            j += 1
        return start, j

    # Generic fallback (not used in your requested behavior, but kept sane)
    new_card = re.compile(
        r'^\s*(CELL_PARAMETERS|ATOMIC_POSITIONS|ATOMIC_SPECIES|K_POINTS|'
        r'OCCUPATIONS|CONSTRAINTS|HUBBARD|BEGIN|END|&control|&system|'
        r'&electrons|&ions|&cell)\b',
        re.IGNORECASE,
    )

    j = start + 1
    while j < len(lines):
        if lines[j].strip() == "" or new_card.match(lines[j]):
            break
        j += 1

    return start, j

import re

# NOTE: allow leading spaces before &system
_SYS = re.compile(r'(?is)^\s*&system\b.*?^\s*/\s*$', re.MULTILINE)

def _prefer_explicit_cell(text: str) -> str:
    def fix(block: str) -> str:
        # 1) ensure ibrav=0 (replace or insert after &system)
        if re.search(r'(?im)^\s*ibrav\s*=', block):
            block = re.sub(
                r'(?im)^(?P<i>\s*)ibrav\s*=\s*[^,/\n]+(?P<t>\s*,?)\s*$',
                r"\g<i>ibrav=  0\g<t>", block, count=1
            )
        else:
            block = re.sub(
                r'(?im)^\s*&system\b\s*$',
                "&system\n  ibrav=  0,", block, count=1
            )

        # 2) remove lattice-parameter tokens anywhere on the line
        block = re.sub(r'(?i)\bcelldm\(\s*[1-6]\s*\)\s*=\s*[^,/\n]+,?\s*', '', block)
        block = re.sub(r'(?i)\b(?:A|B|C|cosAB|cosAC|cosBC)\s*=\s*[^,/\n]+,?\s*', '', block)

        # 3) tidy punctuation/spacing
        while True:
            nb = re.sub(r',\s*,', ',', block)
            if nb == block: break
            block = nb
        block = re.sub(r'(?m)^\s*,\s*', '', block)                       # leading commas
        block = re.sub(r',\s*(\n\s*/\s*$)', r'\1', block)                 # comma before '/'
        block = re.sub(r'[ \t]+', ' ', block)                             # collapse spaces
        return block

    return _SYS.sub(lambda m: fix(m.group(0)), text)


def _strip_trailing_blank_lines(lines: list[str]) -> list[str]:
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines

def update_qe_input(
    text: str,
    cell: Sequence[Sequence[float]] | None = None,
    positions: Sequence[Sequence[float]] | None = None,
    symbols: Sequence[str] | None = None,
    prefix: str | None = None,
    pseudo_dir: str | None = None,
    outdir: str | None = None,
) -> str:
    """
    Update a Quantum ESPRESSO input script.

    Parameters
    ----------
    text : str
        Full QE input file contents (may already contain CELL_PARAMETERS /
        ATOMIC_POSITIONS / prefix / outdir).
    cell : (3,3) array-like, optional
        If provided, replaces (or appends) the CELL_PARAMETERS card.
    positions : (nat,3) array-like, optional
        Cartesian positions. Must be given together with `symbols`.
    symbols : (nat,) sequence of str, optional
        Atomic symbols, same length as `positions`.
    prefix : str, optional
        If provided, replaces existing `prefix = '...'` line
        (first match, case-insensitive). If not found, tries to
        insert inside &control before the closing "/".
    pseudo_dir : str, optional
        If provided, replaces existing `pseudo_dir = '...'` line
        (first match, case-insensitive). If not found, tries to
        insert inside &control before the closing "/".
    outdir : str, optional
        If provided, replaces existing `outdir = '...'` line
        (first match, case-insensitive). If not found, tries to
        insert inside &control before the closing "/".

    Returns
    -------
    str
        Updated QE input text.
    """
    # If no modifications requested, return input verbatim (preserve whitespace)
    if (
        cell is None
        and positions is None
        and symbols is None
        and prefix is None
        and pseudo_dir is None
        and outdir is None
    ):
        return text

    # --- 1) Work line-wise for cards ---
    lines = text.splitlines()

    # CELL_PARAMETERS
    if cell is not None:
        cell_lines = _format_cell_block(cell)
        start, end = _find_card_block(
            lines,
            re.compile(r'^\s*CELL_PARAMETERS\b', re.IGNORECASE),
            n_body_lines_hint=3,
        )
        if start is not None:
            lines = lines[:start] + cell_lines + lines[end:]
        else:
            # Append if missing, but do NOT add an extra blank line
            lines = _strip_trailing_blank_lines(lines)
            lines.extend(cell_lines)

    # ATOMIC_POSITIONS
    if (positions is not None) ^ (symbols is not None):
        raise ValueError("Both positions and symbols must be provided together.")
    if positions is not None and symbols is not None:
        nat = len(symbols)
        pos_lines = _format_positions_block(symbols, positions)
        start, end = _find_card_block(
            lines,
            re.compile(r'^\s*ATOMIC_POSITIONS\b', re.IGNORECASE),
            n_body_lines_hint=nat,
        )
        if start is not None:
            lines = lines[:start] + pos_lines + lines[end:]
        else:
            lines = _strip_trailing_blank_lines(lines)
            lines.extend(pos_lines)

    # Re-join after structural edits
    result = "\n".join(lines)

    # If we provided an explicit cell (or positions), make sure &system
    # does not declare lattice parameters that conflict with ibrav=0 + CELL_PARAMETERS.
    if cell is not None:
        result = _prefer_explicit_cell(result)

    # --- 2) prefix update ---
    if prefix is not None:
        # Match existing prefix line
        pattern = re.compile(
            r'(?im)^(?P<i>\s*)prefix\s*=\s*([\'"]).*?\2(?P<t>\s*,?\s*)$'
        )

        if pattern.search(result):
            def _repl(m):
                return f"{m.group('i')}prefix='{prefix}'{m.group('t')}"
            result = pattern.sub(_repl, result, count=1)
        else:
            # Try inserting into &control before '/'
            ctrl_pat = re.compile(r'(?ims)^&control\b(.*?)/')
            m = ctrl_pat.search(result)
            if m:
                block = m.group(0)
                # insert before final '/'
                new_block = re.sub(
                    r'(?m)^/\s*$',
                    f"    prefix='{prefix}',\n/",
                    block,
                    count=1,
                )
                result = result[:m.start()] + new_block + result[m.end():]
            # else: silently skip if no &control and no prefix line

    # --- 3) pseudo_dir update ---
    if pseudo_dir is not None:
        pattern = re.compile(
            r'(?im)^(?P<i>\s*)pseudo_dir\s*=\s*([\'"]).*?\2(?P<t>\s*,?\s*)$'
        )

        if pattern.search(result):
            def _repl(m):
                return f"{m.group('i')}pseudo_dir='{pseudo_dir}'{m.group('t')}"
            result = pattern.sub(_repl, result, count=1)
        else:
            # Try inserting into &control before '/'
            ctrl_pat = re.compile(r'(?ims)^&control\b(.*?)/')
            m = ctrl_pat.search(result)
            if m:
                block = m.group(0)
                new_block = re.sub(
                    r'(?m)^/\s*$',
                    f"    pseudo_dir='{pseudo_dir}',\n/",
                    block,
                    count=1,
                )
                result = result[:m.start()] + new_block + result[m.end():]
            # else: silently skip

    # --- 4) outdir update ---
    if outdir is not None:
        pattern = re.compile(
            r'(?im)^(?P<i>\s*)outdir\s*=\s*([\'"]).*?\2(?P<t>\s*,?\s*)$'
        )

        if pattern.search(result):
            def _repl(m):
                return f"{m.group('i')}outdir='{outdir}'{m.group('t')}"
            result = pattern.sub(_repl, result, count=1)
        else:
            # Try inserting into &control before '/'
            ctrl_pat = re.compile(r'(?ims)^&control\b(.*?)/')
            m = ctrl_pat.search(result)
            if m:
                block = m.group(0)
                new_block = re.sub(
                    r'(?m)^/\s*$',
                    f"    outdir='{outdir}',\n/",
                    block,
                    count=1,
                )
                result = result[:m.start()] + new_block + result[m.end():]
            # else: silently skip

    return result

