#!/usr/bin/env python3
from itertools import dropwhile
from pathlib import Path
import sys


def main():
    inputs = filter(lambda x: x.strip().endswith(".xpirbc"), sys.argv)
    src = next(inputs)
    outputs_iter = dropwhile(lambda x: x.strip() != "-o", sys.argv)
    next(outputs_iter)
    output = Path(next(outputs_iter))
    print(f"Compiling {src} to {output} via direct copy...")
    output.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
