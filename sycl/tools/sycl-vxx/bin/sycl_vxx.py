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

from argparse import ArgumentError, ArgumentParser
from itertools import starmap
import functools
import json
from multiprocessing import Pool
from os import environ
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tabnanny import check
import tempfile

# This pipeline should be able to do any promotion O3 is capable of
# and some more control flow optimizations then strictly necessary.
# Some more minimization is possible
VXX_PassPipeline = [
"-preparesycl",
"-lower-expect",
"-simplifycfg",
"-sroa",
"-early-cse",
"-annotation2metadata",
"-callsite-splitting",
"-ipsccp",
"-called-value-propagation",
"-globalopt",
"-mem2reg",
"-deadargelim",
"-simplifycfg",
"-inline",
"-function-attrs",
"-argpromotion",
"-sroa",
"-early-cse-memssa",
"-speculative-execution",
"-jump-threading",
"-correlated-propagation",
"-simplifycfg",
"-libcalls-shrinkwrap",
"-tailcallelim",
"-simplifycfg",
"-reassociate",
"-loop-simplify",
"-lcssa",
"-licm",
"-loop-rotate",
"-licm",
"-loop-unswitch",
"-simplifycfg",
"-loop-simplify",
"-lcssa",
"-loop-idiom",
"-indvars",
"-loop-deletion",
"-loop-unroll",
"-sroa",
"-mldst-motion",
"-gvn",
"-sccp",
"-bdce",
"-jump-threading",
"-correlated-propagation",
"-adce",
"-memcpyopt",
"-dse",
"-loop-simplify",
"-lcssa",
"-simplifycfg",
"-elim-avail-extern",
"-rpo-function-attrs",
"-globalopt",
"-globaldce",
"-float2int",
"-lower-constant-intrinsics",
"-loop-simplify",
"-lcssa",
"-loop-rotate",
"-loop-simplify",
"-loop-load-elim",
"-simplifycfg",
"-loop-simplify",
"-lcssa",
"-loop-unroll",
"-loop-simplify",
"-lcssa",
"-licm",
"-alignment-from-assumptions",
"-strip-dead-prototypes",
"-globaldce",
"-constmerge",
"-loop-simplify",
"-lcssa",
"-loop-sink",
"-instsimplify",
"-div-rem-pairs",
"-simplifycfg",
]

class TmpDirManager:
    """ Context handler for a temporary repository that can be programmed
        to be cleaned up when the manager is destroyed.
    """

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


def subprocess_error_handler(msg: str):
    """ Build decorator that prints an error message and prevents
        CompilationDriver from continuing when a called subprocess
        exits with non-zero status
    """
    def decorator(func):
        def decorated(self, *args, **kwargs):
            if self.ok:
                try:
                    return func(self, *args, **kwargs)
                except subprocess.CalledProcessError:
                    print(msg, file=sys.stderr)
                    self.ok = False
        return decorated
    return decorator

def run_if_ok():
    """ Function only runs if the internal ok state is true"""
    def decorator(func):
        def decorated(self, *args, **kwargs):
            if self.ok:
                return func(self, *args, **kwargs)
        return decorated
    return decorator


def _run_in_isolated_proctree(cmd, *args, **kwargs):
    """ Run a command in isolated process namespace.
    This is necessary to get a clean termination of all v++
    subprocesses in case of program interruption, as v++ subprocess
    handling is strange.
    """
    newcmd = ("unshare",
              "--map-current-user",
              "--pid",
              "--mount-proc",
              "--kill-child",
              *cmd)
    return subprocess.run(cmd, *args, **kwargs)


class VXXVersion:
    def __init__(self, exec_path) -> None:
        cmd = (exec_path, '-v')
        proc_res = _run_in_isolated_proctree(cmd, capture_output=True)
        version_regex = r".*v(?P<major>\d{4})\.(?P<minor>\d).*"
        match = re.match(version_regex,
                         proc_res.stdout.decode('utf-8'),
                         flags=re.DOTALL)
        self.major = int(match['major'])
        self.minor = int(match['minor'])
        print(f"Found Vitis version {self}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def _interface_is_attribute(self):
        return self.major >= 2022

    def get_kernel_prop_opt(self):
        ret = []
        if self._interface_is_attribute():
            ret.append("-sycl-vxx-maxi-attr-encoding")
        return ret


class VXXBinary:
    def __init__(self, execname):
        self.path = shutil.which(execname)
        if self.path is None:
            raise FileNotFoundError
        self.path = self.path.resolve()
        self.version = VXXVersion(self.path)
        if environ.get("XILINX_CLANG_39_BUILD_PATH") is not None:
            self.clang_bin_ = Path(environ["XILINX_CLANG_39_BUILD_PATH"]) / "bin"
        else:
            self.clang_bin_ = (
                vitis_exec.parents[2] /
                "/lnx64/tools/clang-3.9-csynth/bin"
            ).resolve()
            if not (self.clang_bin_ / "llvm-as").is_file():
                self.clang_bin_ = (
                    vitis_exec.parents[3] /
                    f"Vitis_HLS/{self.version}" /
                    "lnx64/tools/clang-3.9-csynth/bin"
                ).resolve()

    @property
    def binary_dir(self):
        return self.path.parent

    @property
    def clang_bin(self):
        return self.clang_bin_


class VitisCompilationDriver:
    def __init__(self, arguments, execname):
        self.outpath = Path(arguments.o)
        self.tmp_root = arguments.tmp_root
        self.clang_path = arguments.clang_path.resolve()
        self.inputs = arguments.inputs
        self.vitisexec = VXXBinary(execname)
        self.vitis_bin_dir = self.vitisexec.binary_dir
        self.vitis_version = self.vitisexec.version
        self.outstem = self.outpath.stem
        self.ok = True
        self.vitis_clang_bin = self.vitisexec.clang_bin
        self.cmd_number = 0

    def _dump_cmd(self, stem, args):
        cmdline = " ".join(map(str, args))
        with (self.tmpdir / f"{self.cmd_number:0>3}-{stem}.cmd").open("w") as f:
            f.write(cmdline + "\n")
            if environ.get("SYCL_VXX_DBG_CMD_DUMP") is not None:
                f.write(f"\nOriginal command list: {args}")
            if environ.get("SYCL_VXX_PRINT_CMD") is not None:
                print("SYCL_VXX_CMD:", cmdline)
        self.cmd_number += 1

    def _next_passes(self, inputs):
        return inputs

    @subprocess_error_handler("Linkage of multiple inputs failed")
    def _link_multi_inputs(self, inputs):
        """Link all input files into a single .bc"""
        output = self.tmpdir / f"{self.outstem}-before-opt.bc"
        if len(inputs) > 1:
            llvm_link = self.clang_path / "llvm-link"
            args = [str(llvm_link),
                    *inputs,
                    "-o",
                    str(output)
                    ]
            self._dump_cmd("link_multi_inputs", args)
            subprocess.run(args, check=True)
        else:
            shutil.copy2(inputs[0], output)
        return output

    @subprocess_error_handler("Error in sycl->HLS conversion")
    def _run_preparation(self, inputs):
        """Run the various sycl->HLS conversion passes"""
        # We try to avoid as many optimization as possible
        # to give vitis the opportunity to use its custom
        # optimizations
        outstem = self.outstem
        prepared_bc = (
            self.tmpdir /
            f"{outstem}-kernels-prepared.bc"
        )
        opt_options = [
            "--sycl-vxx",
            "-preparesycl", "-globaldce",]
        opt_options.extend(VXX_PassPipeline)
        opt_options.extend(["-inSPIRation", "-o", f"{prepared_bc}"])

        opt = self.clang_path / "opt"
        args = [opt, *opt_options, inputs]
        self._dump_cmd("run_preparation", args)
        proc = subprocess.run(args, check=True, capture_output=True)
        if bytes("SYCL_VXX_UNSUPPORTED_SPIR_BUILTINS", "ascii") in proc.stderr:
            print("Unsupported SPIR builtins found : stopping compilation")
            self.ok = False
        return prepared_bc

    @subprocess_error_handler("Error when linking with HLS SPIR library")
    def _link_spir(self, inputs):
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
        linked_kernels = self.tmpdir / f"{self.outstem}_kernels-linked.bc"
        args = [
            llvm_link,
            inputs,
            "--only-needed",
            vitis_lib_spir,
            "-o",
            linked_kernels
        ]
        self._dump_cmd("link_spir", args)
        subprocess.run(args, check=True)
        return linked_kernels

    @subprocess_error_handler("Error in preparing and downgrading IR")
    def _downgrade(self, inputs):
        opt = self.clang_path / "opt"
        prepared_kernels = self.tmpdir / f"{self.outstem}_linked.simple.bc"
        kernel_prop = (
            self.tmpdir /
            f"{self.outstem}-kernels_properties.json"
        )

        kernel_prop_opt = ["-kernelPropGen",
                           "--sycl-kernel-propgen-output", f"{kernel_prop}"]

        kernel_prop_opt.extend(self.vitis_version.get_kernel_prop_opt())

        opt_options = [
            "--lower-delayed-sycl-metadata", "-lower-sycl-metadata",
            "--sycl-vxx", "--sycl-prepare-clearspir", "-S", "-preparesycl",
            *kernel_prop_opt,
            "-globaldce",
            "-strip-debug",
            inputs,
            "-o", prepared_kernels
        ]
        args = [opt, *opt_options]
        self._dump_cmd("prepare", args)
        subprocess.run(args, check=True)
        with kernel_prop.open('r') as kp_fp:
            self.kernel_properties = json.load(kp_fp)
        opt_options = ["--sycl-vxx", "-S", "-vxxIRDowngrader"]
        downgraded_ir = (
            self.tmpdir / f"{self.outstem}_kernels-linked.opt.ll")
        args = [
            opt, *opt_options, prepared_kernels,
            "-o", downgraded_ir
        ]
        self._dump_cmd("downgrade", args)
        subprocess.run(args, check=True)
        return downgraded_ir

    @subprocess_error_handler("Downgrading of llvm IR -> Vitis old llvm bitcode failed")
    def _asm_ir(self, inputs):
        """Assemble downgraded IR to bitcode using Vitis llvm-as"""
        vpp_llvm_input = self.tmpdir / f"{self.outstem}_kernels.opt.xpirbc"
        args = [
            self.vitis_clang_bin / "llvm-as",
            inputs,
            "-o",
            vpp_llvm_input
        ]
        self._dump_cmd("05-asm_ir.cmd", args)
        subprocess.run(args, check=True)
        return vpp_llvm_input

    def drive_compilation(self):
        autodelete = environ.get("SYCL_VXX_KEEP_CLUTTER") is None
        outstem = self.outstem
        tmp_root = self.tmp_root
        tmp_manager = TmpDirManager(tmp_root, outstem, autodelete)
        with tmp_manager as self.tmpdir:
            tmpdir = self.tmpdir
            if not autodelete:
                print(f"Temporary clutter in {tmpdir} will not be deleted")
            joined_kernels = self._link_multi_inputs(self.inputs)
            prepared_bc = self._run_preparation(joined_kernels)
            spir_linked = self._link_spir(prepared_bc)
            downgraded = self._downgrade(spir_linked)
            if environ.get("SYCL_VXX_MANUAL_EDIT") is not None:
                print("Please edit", self.downgraded_ir)
                input("Press enter to resume the compilation")
            self.vpp_llvm_input = (
            )
            assembled = self._asm_ir(downgraded)
            final = self._next_passes(assembled)
            try:
                shutil.copy2(final, self.outpath)
            except FileNotFoundError:
                print(
                    f"Output {self.xclbin} was not properly produced by previous commands")
            return self.ok


class VXXCompilationDriver(VitisCompilationDriver):
    def __init__(self, arguments):
        super().__init__(arguments, "v++")
        self.vitis_mode = arguments.target
        # TODO: XILINX_PLATFORM should be passed by clang driver instead
        self.xilinx_platform = environ['XILINX_PLATFORM']
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
        if self.hls_flow and (self.vitis_mode == "sw_emu"):
            raise Exception("sw_emu is not compatible with the HLS flow")

    @run_if_ok
    def _get_compile_kernel_cmd_out(self, kernel, inputs):
        """Create command to compile kernel"""
        vxx = self.vitis_bin_dir / "v++"
        comp_config = environ.get('SYCL_VXX_COMP_CONFIG')
        kernel_output = self.tmpdir / f"{kernel['name']}.xo"
        command = [
            vxx, "--target", self.vitis_mode,
            "--advanced.param", "compiler.hlsDataflowStrictMode=off",
            # Do the optimizations that were not performed by sycl compiler
            "-O3",
            "--platform", self.xilinx_platform,
            "--temp_dir", self.tmpdir / 'vxx_comp_tmp',
            "--log_dir", self.tmpdir / 'vxx_comp_log',
            "--report_dir", self.tmpdir / 'vxx_comp_report',
            "--save-temps", "-c", "-k", kernel['name'], '-o', kernel_output,
            inputs
        ]
        if comp_config is not None and Path(comp_config).is_file():
            command.extend(("--config", Path(comp_config).resolve()))
        if 'extra_args' in kernel and kernel['extra_args'].strip():
            # User provided kernel arguments can contain many spaces,
            # leading split to give empty string that are incorrectly
            # interpreted as file name by v++ : filter remove them
            command.extend(
                filter(lambda x: x != '', kernel['extra_args'].split(' ')))
        command.extend(self.extra_comp_args)
        self._dump_cmd(f"vxxcomp-{kernel['name']}", command)
        return (kernel_output, command)

    @run_if_ok
    def _compile_kernel(self, outname, command):
        """Execute a kernel compilation command"""
        _run_in_isolated_proctree(command, check=True)
        return outname

    @subprocess_error_handler("Vitis linkage stage failed")
    def _link_kernels(self):
        """Call v++ to link all kernel in one .xclbin"""
        xclbin = self.tmpdir / f"{self.outstem}.xclbin"
        vpp = self.vitis_bin_dir / "v++"
        link_config = environ.get('SYCL_VXX_LINK_CONFIG')
        command = [
            vpp, "--target", self.vitis_mode,
            "--advanced.param", "compiler.hlsDataflowStrictMode=off",
            "--platform", self.xilinx_platform,
            "--temp_dir", self.tmpdir / 'vxx_link_tmp',
            "--log_dir", self.tmpdir / 'vxx_link_log',
            "--report_dir", self.tmpdir / 'vxx_link_report',
            "--save-temps", "-l", "-o", xclbin
        ]
        if link_config is not None and Path(link_config).is_file():
            command.extend(("--config", Path(link_config).resolve()))
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
        self._dump_cmd("vxxlink", command)
        _run_in_isolated_proctree(command, check=True)
        return xclbin

    @subprocess_error_handler("Vitis compilation stage failed")
    def _launch_parallel_compilation(self, inputs):
        # Compilation commands are generated in main process to ensure
        # they are printed on main process stdout if command dump is set
        compile_commands = map(
            functools.partial(self._get_compile_kernel_cmd_out, inputs=inputs),
            self.kernel_properties["kernels"])
        if environ.get("SYCL_VXX_SERIALIZE_VITIS_COMP") is None:
            p = Pool()
            try:
                future = p.starmap_async(
                    self._compile_kernel,
                    compile_commands)
                compiled_kernels = list(future.get())
            except KeyboardInterrupt:
                p.terminate()
                raise KeyboardInterrupt
        else:
            self.compiled_kernels = list(starmap(self._compile_kernel, compile_commands))

    def _next_passes(self, inputs):
        # Driver specific area
        kernels = self._launch_parallel_compilation(inputs)
        xclbin = self._link_kernels(kernels)
        return xclbin

class IPExportCompilationDriver(VitisCompilationDriver):
    def __init__(self, arguments):
        super().__init__(arguments, "vitis_hls")
        self.target == arguments.target
        self.clock_period = arguments.clock_period
    
    @run_if_ok
    def _get_top_comp_name(self):
        numKernels = len(self.kernel_properties["kernels"])
        if numKernels != 1:
            raise Exception(
                f"{numKernels} top level components found, should be exactly one")
        kernelname = self.kernel_properties["kernels"][0]['name']
        return kernelname

    @run_if_ok
    def _create_hls_script(self, compname, inputs):
        script = self.tmpdir / "run_hls.tcl"
        out = self.tmpdir / f"{compname}.zip"
        with script.open("w") as sf:
            sf.writelines([
                "open_project -reset proj",
                f"add_files {inputs}",
                f"set_top {compname}",
                "open_solution -reset sol",
                f"set_part {self.target}",
                f"create_clock -period {self.clock_period} -name default"
                "csynth_design"
                f"export_design -format ip_catalog -output {out}"
                ])
        return script, out
    
    @subprocess_error_handler("Vitis HLS invocation failed")
    def _run_vitis_hls(self, script):
        cmd = (self.vitisexec.path, script)
        self._dump_cmd("vitis-hls-invocation", cmd)
        _run_in_isolated_proctree(cmd, check=True)

    def _next_passes(self, inputs):
        topcompname = self._get_top_comp_name()
        script, final_comp = self._create_hls_script(topcompname, inputs)
        self._run_vitis_hls(script)
        return final_comp


def parse_args(args=sys.argv[1:]):
    parser = ArgumentParser(
        description="Utility to drive various compilation flow vor vivado related tools")
    command = args[1]
    if command == "vxxcompile":
        parser.add_argument(
            "--hls",
            help="Activate the hls flow instead of the default spir one",
            action="store_true")
        parser.add_argument(
            "--target",
            help="v++ synthesis mode",
            choices=["sw_emu", "hw_emu", "hw"],
            required=True)
        parser.add_argument(
            "--vitis_comp_argfile",
            help="file containing v++ -c argument",
            type=Path)
        parser.add_argument(
            "--vitis_link_argfile",
            help="file containing v++ -l argument",
            type=Path)
    elif command == "ipexport":
        parser.add_argument(
            "--target",
            help="Part code for which the synthesis should be done",
            required=True)
        # TODO delay the default to clang driver, make it required here 
        parser.add_argument(
            "--clock-period",
            help="clock period description",
            default="3ns"
        )
    else:
        raise ArgumentError(message=f"Unknown command: {command}")

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

    parser.add_argument("-o", help="output file name", required=True)
    parser.add_argument("inputs", nargs="+")
    return command, parser.parse_args(args=args)


def main():
    """Script entry function"""
    command, args = parse_args()
    if command == "vxxcompile":
        cd = VXXCompilationDriver(args)
    else:
        cd = IPExportCompilationDriver(args)
    return cd.drive_compilation()


if __name__ == "__main__":
    import sys
    try:
        if (not main()):
            sys.exit(-1)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, will stop")
        sys.exit(-2)
