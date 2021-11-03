#!/usr/bin/env python3
"""sycl_vxx_post_link
This is the sycl-post-link implementation for the VXX toolchain which is able
to get kernel names from xclbins (Xilinx FPGA native binaries).
"""

from argparse import ArgumentParser
import subprocess
from pathlib import Path


class PostLinkEntriesExtractor:
    def __init__(self, args) -> None:
        self.output = args.o
        self.inputs = args.inputs

    def extract(self):
        def get_kernels(xclbin):
            process = subprocess.run(["xclbinutil", "-i", xclbin,
                                      "--info"], capture_output=True)
            for line in process.stdout.decode("utf8").split("\n"):
                if line.strip().startswith("Kernels:"):
                    line = line.replace("Kernels:", "", 1).strip()
                    kernels = {s.strip() for s in line.split(',')}
                    return kernels
            raise EOFError("No kernel found")

        def handle_input(input_n: int, filename: Path, outfile):
            outdir = self.output.resolve().parent
            stem = self.output.stem
            #outprops = outdir / f'{stem}_{input_n}.props'
            outsyms = outdir / f'{stem}_{input_n}.sym'
            #outprops.touch()
            with outsyms.open("w") as symfile:
                symfile.write("\n".join(get_kernels(filename)))
                symfile.write("\n")
            outfile.write(f"{filename}|{outsyms}\n")

        with open(self.output, "w") as outfile:
            outfile.write("[Code|Symbols]\n")
            for (i, filename) in enumerate(self.inputs):
                handle_input(i, filename, outfile)
        return True


def main():
    """Script entry function"""
    parser = ArgumentParser()
    parser.add_argument("-o", help="output name", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    cd = PostLinkEntriesExtractor(args)
    return cd.extract()


if __name__ == "__main__":
    import sys
    try:
        if (not main()):
            sys.exit(-1)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, will stop")
        sys.exit(-2)
