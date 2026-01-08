from typing import Tuple

def _parse_triplet_bools(s: str) -> Tuple[bool, bool, bool]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated values like '1,1,1'")
    out = []
    for p in parts:
        pl = p.lower()
        if pl in ("t", "true", "1", "y", "yes"):
            out.append(True)
        elif pl in ("f", "false", "0", "n", "no"):
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean: {p!r}")
    return (out[0], out[1], out[2])


def _parse_float_list(s: str) -> list[float]:
    xs = [x.strip() for x in s.split(",") if x.strip()]
    if not xs:
        raise argparse.ArgumentTypeError("Expected comma-separated floats, e.g. '1.23,4.56'")
    try:
        return [float(x) for x in xs]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


