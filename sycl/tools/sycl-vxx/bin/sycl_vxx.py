#!/usr/bin/env python3
"""sycl_vxx
This is an extra layer of abstraction on top of the shell invoking the
Vitis compiler. As SPIR/LLVM-IR is a second class citizen in Vitis for the
moment it has some little niggling details that need worked on and the idea
is that this script will work around those with some aid from Clang/LLVM.

One of the main examples is that Vitis can only compile one kernel from LLVM-BC
at a time and it requires the kernel name (also required for kernel-specific
optimizations). This poses a problem as there can be multiple kernels in a
file. And when making a normal naked v++ -c command in the driver,
you won't have the necessary information as the command is generated before
the file has have even started to be compiled (perhaps there is, but I am
unaware of). So, no kernel name and no idea how many Vitis commands you'd need
to generate per file (no idea how many kernels are in a file).

This works around that by using an opt (kernelNameGen) pass that generates
an intermediate file with the needed information that we eat up and can then
loop over each kernel in a file. It's simple at the moment: just kernel names,
but could expand in the future to include optimization information for each
kernel.
"""

from argparse import ArgumentParser
import json
from multiprocessing import Pool
from os import environ
from pathlib import Path
import shutil
import subprocess
import tempfile


class TmpDirManager:
    def __init__(self, tmpdir: Path, prefix: str, autodelete: bool):
        self.prefix = prefix
        self.tmpdir = tmpdir
        self.autodelete = autodelete

    def __enter__(self) -> Path:
        self.dir = Path(tempfile.mkdtemp(
            dir=self.tmpdir,
            prefix=self.prefix
        ))
        return self.dir

    def __exit__(self, *_):
        if (self.autodelete):
            shutil.rmtree(self.dir)


class CompilationDriver:
    def __init__(self, arguments):
        self.outpath = Path(arguments.o)
        self.tmp_root = arguments.tmp_root
        self.clang_path = arguments.clang_path.resolve()
        self.inputs = arguments.inputs
        self.hls_flow = arguments.hls
        self.vitis_bin_dir = arguments.vitis_bin_dir.resolve()
        self.vitis_version = self.vitis_bin_dir.parent.name
        self.outstem = self.outpath.stem
        self.vitis_mode = arguments.target
        # TODO: XILINX_PLATFORM should be passed by clang driver instead
        self.xilinx_platform = environ['XILINX_PLATFORM']
        if environ.get("XILINX_CLANG_39_BUILD_PATH") is not None:
            vitis_clang_bin = Path(environ["XILINX_CLANG_39_BUILD_PATH"])
        else:
            vitis_clang_bin = (
                self.vitis_bin_dir.parent /
                "/lnx64/tools/clang-3.9-csynth/bin"
            ).resolve()
            if not (vitis_clang_bin / "llvm-as").is_file():
                vitis_clang_bin = (
                    self.vitis_bin_dir.parents[2] /
                    f"Vitis_HLS/{self.vitis_version}" /
                    "lnx64/tools/clang-3.9-csynth/bin"
                ).resolve()
        self.vitis_clang_bin = vitis_clang_bin
        self.extra_comp_args = []
        if arguments.vitis_comp_argfile is not None:
            with arguments.vitis_comp_argfile.open("r") as f:
                content = f.read().strip()
                if content:
                    self.extra_comp_args.extend(content.split(' '))
        self.extra_link_args = []
        if arguments.vitis_link_argfile is not None:
            with arguments.vitis_link_argfile.open("r") as f:
                content = f.read().strip()
                if content:
                    self.extra_link_args.extend(content.split(' '))

    def _dump_cmd(self, filename, args):
        with (self.tmpdir / filename).open("w") as f:
            f.write(" ".join(map(str, args)) + "\n")
            if environ.get("SYCL_VXX_DBG_CMD_DUMP") is not None:
                f.write(f"\nOriginal command list: {args}")

    def _link_multi_inputs(self):
        """Link all input files into a single .bc"""
        llvm_link = self.clang_path / "llvm-link"
        args = [str(llvm_link),
                *self.inputs,
                "-o",
                str(self.before_opt_src)
                ]
        self._dump_cmd("00-link_multi_inputs.cmd", args)
        subprocess.run(args)

    def _run_optimisation(self):
        """Run the various sycl->HLS conversion passes"""
        outstem = self.outstem
        kernel_prop = (
            self.tmpdir /
            f"{outstem}-kernels_properties.json"
        )
        self.optimised_bc = (
            self.tmpdir /
            f"{outstem}-kernels-optimized.bc"
        )
        opt_options = ["--sycl-vxx",
                       "-kernelPropGen",
                       "--sycl-kernel-propgen-output", f"{kernel_prop}",
                       "-preparesycl", "-globaldce"]
        if not self.hls_flow:
            opt_options.extend([
                "-inline", "-infer-address-spaces",
                "-flat-address-space=0", "-globaldce"
            ])
        opt_options.extend([
            "-O3", "-globaldce", "-globaldce", "-inSPIRation",
            "-o", f"{self.optimised_bc}"
        ])

        opt = self.clang_path / "opt"
        args = [opt, *opt_options, self.before_opt_src]
        self._dump_cmd("01-run_optimisations.cmd", args)
        subprocess.run(args)
        with kernel_prop.open('r') as kp_fp:
            self.kernel_properties = json.load(kp_fp)

    def _link_spir(self):
        vitis_lib_spir = (
            self.vitis_bin_dir.parent /
            "lnx64/lib/libspir64-39-hls.bc"
        ).resolve()
        if not vitis_lib_spir.is_file():
            vitis_lib_spir = (
                self.vitis_bin_dir.parents[2] /
                f"Vitis_HLS/{self.vitis_version}/lnx64/lib/libspir64-39-hls.bc"
            ).resolve()
        llvm_link = self.clang_path / 'llvm-link'
        self.linked_kernels = self.tmpdir / f"{self.outstem}_kernels-linked.bc"
        args = [
            llvm_link,
            self.optimised_bc,
            "--only-needed",
            vitis_lib_spir,
            "-o",
            self.linked_kernels
        ]
        self._dump_cmd("02-link_spir.cmd", args)
        subprocess.run(args)

    def _prepare_and_downgrade(self):
        opt = self.clang_path / "opt"
        prepared_kernels = self.tmpdir / f"{self.outstem}_linked.simple.bc"
        opt_options = [
            "--sycl-vxx", "--sycl-prepare-clearspir", "-S", "-preparesycl",
            "-globaldce", self.linked_kernels,
            "-o", prepared_kernels
        ]
        args = [opt, *opt_options]
        self._dump_cmd("03-prepare.cmd", args)
        subprocess.run(args)
        opt_options = ["--sycl-vxx", "-S", "-O3", "-vxxIRDowngrader"]
        self.downgraded_ir = (
            self.tmpdir / f"{self.outstem}_kernels-linked.opt.ll")
        args = [
            opt, *opt_options, prepared_kernels,
            "-o", self.downgraded_ir
        ]
        self._dump_cmd("04-downgrade.cmd", args)
        subprocess.run(args)

    def _asm_ir(self):
        """Assemble downgraded IR to bitcode using Vitis llvm-as"""
        args = [
            self.vitis_clang_bin / "llvm-as",
            self.downgraded_ir,
            "-o",
            self.vpp_llvm_input
        ]
        self._dump_cmd("05-asm_ir.cmd", args)
        subprocess.run(args)
        if self.hls_flow and self.vitis_mode == "sw_emu":
            # assemble the xpirbc due to bug in v++ when provided with llvm IR
            proc = subprocess.run([
                self.vitis_clang_bin / 'llvm-config',
                '--host-target'
            ], capture_output=True)
            host_triple = proc.stdout.decode('utf-8')
            self.assembled_xpirbc = self.tmpdir / f"{self.outstem}.S"
            self.host_native_code = self.tmpdir / f"{self.outstem}.o"
            subprocess.run([
                self.vitis_clang_bin / "llc", f"-mtriple={host_triple}",
                "-o", self.assembled_xpirbc,
                self.vpp_llvm_input
            ])
            subprocess.run([
                self.vitis_clang_bin / "llc", f"-mtriple={host_triple}",
                "-o", self.host_native_code,
                "--filetype=obj",
                self.vpp_llvm_input
            ])

    def _compile_kernel(self, kernel):
        """Generate .xo from kernel"""
        vxx = self.vitis_bin_dir / "v++"
        kernel_output = self.tmpdir / f"{kernel['name']}.xo"
        command = [
            vxx, "--target", self.vitis_mode,
            "--advanced.param", "compiler.hlsDataflowStrictMode=off",
            "--platform", self.xilinx_platform,
            "--temp_dir", self.tmpdir / 'vxx_comp_tmp',
            "--log_dir", self.tmpdir / 'vxx_comp_log',
            "--report_dir", self.tmpdir / 'vxx_comp_report',
            "--save-temps", "-c", "-k", kernel['name'], '-o', kernel_output,
            self.vpp_llvm_input
        ]
        if 'extra_args' in kernel and kernel['extra_args'].strip():
            # User provided kernel arguments can contain many spaces,
            # leading split to give empty string that are incorrectly
            # interpreted as file name by v++ : filter remove them
            command.extend(
                filter(lambda x: x != '', kernel['extra_args'].split(' ')))
        command.extend(self.extra_comp_args)
        self._dump_cmd(f"06-vxxcomp-{kernel['name']}.cmd", command)
        subprocess.run(command)
        return kernel_output

    def _link_kernels(self):
        """Call v++ to link all kernel in one .xclbin"""
        vpp = self.vitis_bin_dir / "v++"
        command = [
            vpp, "--target", self.vitis_mode,
            "--advanced.param", "compiler.hlsDataflowStrictMode=off",
            "--platform", self.xilinx_platform,
            "--temp_dir", self.tmpdir / 'vxx_link_tmp',
            "--log_dir", self.tmpdir / 'vxx_link_log',
            "--report_dir", self.tmpdir / 'vxx_link_report',
            "--save-temps", "-l", "-o", self.outpath
        ]
        for kernelprop in self.kernel_properties['kernels']:
            targets = dict()
            for mem_assign in kernelprop["bundle_hw_mapping"]:
                command.extend((
                    "--connectivity.sp",
                    "{}_1.m_axi_{}:{}".format(
                        kernelprop["name"],
                        mem_assign["maxi_bundle_name"],
                        mem_assign["target_bank"]
                    )
                ))
                targets[mem_assign["maxi_bundle_name"]
                        ] = mem_assign["target_bank"]
            for arg_assign in kernelprop["arg_bundle_mapping"]:
                arg_name = arg_assign["arg_name"]
                target = targets[arg_assign["maxi_bundle_name"]]
                command.extend((
                    "--connectivity.sp",
                    "{}_1.{}:{}".format(
                        kernelprop["name"],
                        arg_name,
                        target
                    )
                ))
        command.extend(self.extra_link_args)
        command.extend(self.compiled_kernels)
        self._dump_cmd("07-vxxlink.cmd", command)
        subprocess.run(command)

    def drive_compilation(self):
        if self.hls_flow and (self.vitis_mode == "sw_emu"):
            raise Exception("sw_emu is not compatible with the HLS flow")
        autodelete = environ.get("SYCL_VXX_KEEP_CLUTTER") is None
        outstem = self.outstem
        tmp_root = self.tmp_root
        tmp_manager = TmpDirManager(tmp_root, outstem, autodelete)
        with tmp_manager as self.tmpdir:
            tmpdir = self.tmpdir
            if not autodelete:
                print(f"Temporary clutter in {tmpdir} will not be deleted")
            self.before_opt_src = self.tmpdir / f"{outstem}-before-opt.bc"
            if len(self.inputs) > 1:
                self._link_multi_inputs()
            else:
                shutil.copy2(self.inputs[0], self.before_opt_src)
            self._run_optimisation()
            self._link_spir()
            self._prepare_and_downgrade()
            if environ.get("SYCL_VXX_MANUAL_EDIT") is not None:
                print("Please edit", self.downgraded_ir)
                input("Press enter to resume the compilation")
            self.vpp_llvm_input = (
                tmpdir / f"{outstem}_kernels.opt.xpirbc")
            self._asm_ir()
            with Pool() as p:
                self.compiled_kernels = list(p.map(self._compile_kernel, list(
                    k for k in self.kernel_properties["kernels"])))
            if self.compiled_kernels:
                self._link_kernels()


def main():
    """Script entry function"""
    parser = ArgumentParser()
    parser.add_argument(
        "--hls",
        help="Activate the hls flow instead of the default spir one",
        action="store_true")
    parser.add_argument(
        "--vitis_bin_dir",
        help="Vitis bin directory",
        required=True,
        type=Path)
    parser.add_argument(
        "--clang_path",
        help="path to the clang driver that's executing the script",
        required=True,
        type=Path)
    parser.add_argument(
        "--tmp_root",
        help="The temporary directory where we'll put some intermediate files",
        required=True,
        type=Path)
    parser.add_argument(
        "--target",
        help="v++ synthesis mode",
        choices=["sw_emu", "hw_emu", "hw"],
        required=True)
    parser.add_argument("-o", help="output xclbin name", required=True)
    parser.add_argument(
        "--vitis_comp_argfile",
        help="file containing v++ -c argument",
        type=Path)
    parser.add_argument(
        "--vitis_link_argfile",
        help="file containing v++ -l argument",
        type=Path)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    cd = CompilationDriver(args)
    cd.drive_compilation()


if __name__ == "__main__":
    import sys
    print(sys.argv)
    main()
