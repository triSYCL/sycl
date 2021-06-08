# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile
from distutils.spawn import find_executable

import lit.formats
import lit.util
import getpass
import tempfile

from lit.llvm import llvm_config

def split_target(target):
    if target.startswith("hls_"):
        return ("hls", target[4:])
    else:
        return ("spir", target)

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'SYCL-on-device'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.dump'] #add .spv. Currently not clear what to do with those

# feature tests are considered not so lightweight, so, they are excluded by default
config.excludes = ['Inputs', 'feature-tests', 'disabled', '_x', '.Xil', '.run']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.sycl_obj_root, 'test')

config.environment['SYCL_VXX_KEEP_CLUTTER'] = 'True'

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['PATH', 'OCL_ICD_FILENAMES', 'SYCL_DEVICE_ALLOWLIST', 'SYCL_CONFIG_FILE_NAME'])

timeout=600

xocc=lit_config.params.get('XOCC', "off")
xocc_target = "hls_hw_emu"
if "XOCC_TARGET" in os.environ:
    xocc_target = f"hls_{os.environ['XOCC_TARGET']}"

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add('linux')
    llvm_config.with_system_environment('LD_LIBRARY_PATH')
    llvm_config.with_environment('LD_LIBRARY_PATH', config.sycl_libs_dir, append_path=True)
    llvm_config.with_system_environment('CFLAGS')
    llvm_config.with_environment('CFLAGS', config.sycl_clang_extra_flags)

elif platform.system() == "Windows":
    config.available_features.add('windows')
    llvm_config.with_system_environment('LIB')
    llvm_config.with_environment('LIB', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment('CPATH')
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1", append_path=True)
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/", append_path=True)
    llvm_config.with_environment('DYLD_LIBRARY_PATH', config.sycl_libs_dir)

llvm_config.with_environment('PATH', config.sycl_tools_dir, append_path=True)

config.substitutions.append( ('%threads_lib', config.sycl_threads_lib) )
config.substitutions.append( ('%sycl_libs_dir',  config.sycl_libs_dir ) )
config.substitutions.append( ('%sycl_include',  config.sycl_include ) )
config.substitutions.append( ('%sycl_source_dir', config.sycl_source_dir) )
config.substitutions.append( ('%opencl_libs_dir',  config.opencl_libs_dir) )
config.substitutions.append( ('%opencl_include_dir',  config.opencl_include_dir) )
config.substitutions.append( ('%level_zero_include_dir',  config.level_zero_include_dir) )
config.substitutions.append( ('%cuda_toolkit_include',  config.cuda_toolkit_include) )
config.substitutions.append( ('%sycl_tools_src_dir',  config.sycl_tools_src_dir ) )
config.substitutions.append( ('%llvm_build_lib_dir',  config.llvm_build_lib_dir ) )
config.substitutions.append( ('%llvm_build_bin_dir',  config.llvm_build_bin_dir ) )

if config.level_zero_include_dir:
    config.available_features.add("level_zero_headers")
else:
    lit_config.warning("Level_Zero headers path is not configured. Dependent tests are skipped.")


llvm_config.use_clang()

llvm_config.add_tool_substitutions(['llvm-spirv'], [config.sycl_tools_dir])

backend=lit_config.params.get('SYCL_PLUGIN', "opencl")
lit_config.note("Backend: {}".format(backend))
config.substitutions.append( ('%sycl_be', { 'opencl': 'PI_OPENCL',  'cuda': 'PI_CUDA', 'level_zero': 'PI_LEVEL_ZERO'}[backend]) )
config.substitutions.append( ('%BE_RUN_PLACEHOLDER', "env SYCL_DEVICE_FILTER={SYCL_PLUGIN} ".format(SYCL_PLUGIN=backend)) )
config.substitutions.append( ('%RUN_ON_HOST', "env SYCL_DEVICE_FILTER=host ") )

get_device_count_by_type_path = lit_config.params.get('GET_DEVICE_TOOL', os.path.join(config.llvm_tools_dir, "get_device_count_by_type"))
if 'GET_DEVICE_TOOL' in lit_config.params.keys():
    lit_config.warning("The tool from none-default path is used: "+get_device_count_by_type_path)

def getDeviceCount(device_type, be = backend):
    is_cuda = False;
    is_level_zero = False;
    device_count_env = os.environ.copy()
    process = subprocess.Popen([get_device_count_by_type_path, device_type, be],
        stdout=subprocess.PIPE, env=device_count_env)
    (output, err) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0 and be == backend:
        lit_config.error("getDeviceCount {TYPE} {BACKEND}: Non-zero exit code {CODE}".format(
            TYPE=device_type, BACKEND=be, CODE=exit_code))
        return [0,False,False]

    result = output.decode().replace('\n', '').split(':', 1)
    try:
        value = int(result[0])
    except ValueError:
        value = 0
        lit_config.error("getDeviceCount {TYPE} {BACKEND}: Cannot get value from output: {OUT}".format(
            TYPE=device_type, BACKEND=be, OUT=result[0]))

    # if we have found gpu and there is additional information, let's check
    # whether this is CUDA device or Level Zero device or none of these.
    if device_type == "gpu" and value > 0 and len(result[1]):
        if re.match(r".*cuda", result[1]):
            is_cuda = True;
        if re.match(r".*level zero", result[1]):
            is_level_zero = True;

    if err:
        lit_config.warning("getDeviceCount {TYPE} {BACKEND} stderr:{ERR}".format(
            TYPE=device_type, BACKEND=be, ERR=err))
    return [value,is_cuda,is_level_zero]

# Every SYCL implementation provides a host implementation.
config.available_features.add('host')

# Configure device-specific substitutions based on availability of corresponding
# devices/runtimes

found_at_least_one_device = False

cpu_run_substitute = "true"
cpu_run_on_linux_substitute = "true "
cpu_check_substitute = ""
cpu_check_on_linux_substitute = ""

if getDeviceCount("cpu")[0]:
    found_at_least_one_device = True
    lit_config.note("Found available CPU device")
    cpu_run_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:cpu ".format(SYCL_PLUGIN=backend)
    cpu_check_substitute = "| FileCheck %s"
    config.available_features.add('cpu')
    if platform.system() == "Linux":
        cpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:cpu ".format(SYCL_PLUGIN=backend)
        cpu_check_on_linux_substitute = "| FileCheck %s"
else:
    lit_config.warning("CPU device not found")

config.substitutions.append( ('%CPU_RUN_PLACEHOLDER',  cpu_run_substitute) )
config.substitutions.append( ('%CPU_RUN_ON_LINUX_PLACEHOLDER',  cpu_run_on_linux_substitute) )
config.substitutions.append( ('%CPU_CHECK_PLACEHOLDER',  cpu_check_substitute) )
config.substitutions.append( ('%CPU_CHECK_ON_LINUX_PLACEHOLDER',  cpu_check_on_linux_substitute) )

gpu_run_substitute = "true"
gpu_run_on_linux_substitute = "true "
gpu_check_substitute = ""
gpu_check_on_linux_substitute = ""

cuda = False
level_zero = False
[gpu_count, cuda, level_zero] = getDeviceCount("gpu")

if gpu_count > 0:
    found_at_least_one_device = True
    lit_config.note("Found available GPU device")
    gpu_run_substitute = " env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=backend)
    gpu_check_substitute = "| FileCheck %s"
    config.available_features.add('gpu')
    if cuda:
       config.available_features.add('cuda')
    elif level_zero:
       config.available_features.add('level_zero')

    if platform.system() == "Linux":
        gpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=backend)
        gpu_check_on_linux_substitute = "| FileCheck %s"
    # ESIMD-specific setup. Requires OpenCL for now.
    esimd_run_substitute = " env SYCL_DEVICE_FILTER=opencl:gpu SYCL_PROGRAM_COMPILE_OPTIONS=-vc-codegen"
    config.substitutions.append( ('%ESIMD_RUN_PLACEHOLDER',  esimd_run_substitute) )
    config.substitutions.append( ('%clangxx-esimd',  "clang++ -fsycl-explicit-simd" ) )
else:
    lit_config.warning("GPU device not found")

config.substitutions.append( ('%GPU_RUN_PLACEHOLDER',  gpu_run_substitute) )
config.substitutions.append( ('%GPU_RUN_ON_LINUX_PLACEHOLDER',  gpu_run_on_linux_substitute) )
config.substitutions.append( ('%GPU_CHECK_PLACEHOLDER',  gpu_check_substitute) )
config.substitutions.append( ('%GPU_CHECK_ON_LINUX_PLACEHOLDER',  gpu_check_on_linux_substitute) )

acc_run_substitute = "true"
acc_check_substitute = ""
if getDeviceCount("accelerator")[0]:
    found_at_least_one_device = True
    lit_config.note("Found available accelerator device")
    acc_run_substitute = " env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:acc ".format(SYCL_PLUGIN=backend)
    acc_check_substitute = "| FileCheck %s"
    config.available_features.add('accelerator')
else:
    lit_config.warning("Accelerator device not found")

if xocc != "off":
    # xrt doesn't deal well with multiple executables using it concurrently (at the time of writing).
    # The details are at https://xilinx.github.io/XRT/master/html/multiprocess.html
    # so we wrap every use of XRT inside an file lock.
    # We also wrap invocation of executable in an setsid to prevent
    # a single program failure from ending all the tests.
    xrt_lock = f"{tempfile.gettempdir()}/xrt-{getpass.getuser()}.lock"
    acc_run_substitute+= "setsid flock --exclusive " + xrt_lock + " "
    if os.path.exists(xrt_lock):
        os.remove(xrt_lock)
    acc_run_substitute="env --unset=XCL_EMULATION_MODE " + acc_run_substitute
    # hw_emu is very slow so it has a higher timeout.
    if not xocc_target.endswith("hw_emu"):
        acc_run_substitute+= "timeout 3000 env "
    else:
        acc_run_substitute+= "timeout 3000 env "

config.substitutions.append( ('%ACC_RUN_PLACEHOLDER',  acc_run_substitute) )
config.substitutions.append( ('%ACC_CHECK_PLACEHOLDER',  acc_check_substitute) )

# LIT testing either supports OpenCL or CUDA or Level Zero.
if not cuda and not level_zero and found_at_least_one_device:
    config.available_features.add('opencl')

if cuda:
    config.substitutions.append( ('%sycl_triple',  "nvptx64-nvidia-cuda-sycldevice" ) )
elif xocc != "off":
    config.substitutions.append( ('%sycl_triple',  f"fpga64_{xocc_target}-xilinx-unknown-sycldevice" ) )
else:
    config.substitutions.append( ('%sycl_triple',  "spir64-unknown-linux-sycldevice" ) )

if "opencl-aot" in config.llvm_enable_projects:
    lit_config.note("Using opencl-aot version which is built as part of the project")
    config.available_features.add("opencl-aot")
    llvm_config.add_tool_substitutions(['opencl-aot'], [config.sycl_tools_dir])

# Device AOT compilation tools aren't part of the SYCL project,
# so they need to be pre-installed on the machine
aot_tools = ["ocloc", "aoc"]
if "opencl-aot" not in config.llvm_enable_projects:
    aot_tools.append('opencl-aot')

for aot_tool in aot_tools:
    if find_executable(aot_tool) is not None:
        lit_config.note("Found pre-installed AOT device compiler " + aot_tool)
        config.available_features.add(aot_tool)
    else:
        lit_config.warning("Couldn't find pre-installed AOT device compiler " + aot_tool)

if xocc == "off":
    config.excludes += ['xocc']
else:
    if getDeviceCount("gpu", "cuda")[1]:
        lit_config.note("found secondary cuda target")
        config.available_features.add("has_secondary_cuda")
    lit_config.note("XOCC target: {}".format(xocc_target))
    required_env = ['HOME', 'USER', 'XILINX_XRT', 'XILINX_SDX', 'XILINX_PLATFORM', 'EMCONFIG_PATH', 'LIBRARY_PATH', "XILINX_VITIS"]
    has_error=False
    config.available_features.add("xocc")
    for feature in split_target(xocc_target):
        config.available_features.add(feature) 
    config.available_features.add(xocc_target)
    pkg_opencv4 = subprocess.run(["pkg-config", "--libs", "--cflags", "opencv4"], stdout=subprocess.PIPE)
    has_opencv4 = not pkg_opencv4.returncode
    lit_config.note("has opencv4: {}".format(has_opencv4))
    if has_opencv4:
        config.available_features.add("opencv4")
        config.substitutions.append( ('%opencv4_flags', pkg_opencv4.stdout.decode('utf-8')[:-1]) )
    for env in required_env:
        if env not in os.environ:
            lit_config.note("missing environnement variable: {}".format(env))
            has_error=True
    if has_error:
        lit_config.error("Can't configure tests for XOCC")
    llvm_config.with_system_environment(required_env)
    if xocc == "only":
        config.excludes += ['basic_tests', 'extentions', 'online_compiler', 'plugins']
    # run_if_* defaults to a simple echo to print the comand instead of running it.
    # it will be replaced by and empty string to actually run the command.
    run_if_hw="echo"
    run_if_hw_emu="echo"
    run_if_sw_emu="echo"
    if xocc_target.endswith("_hw"):
        timeout = 10800 # 3h
        run_if_hw=""
    if xocc_target.endswith("_hw_emu"):
        timeout = 3600 # 1h
        run_if_hw_emu=""
    if xocc_target.endswith("_sw_emu"):
        timeout = 1200 # 20min
        run_if_sw_emu=""
    config.substitutions.append( ('%run_if_hw', run_if_hw) )
    config.substitutions.append( ('%run_if_hw_emu', run_if_hw_emu) )
    config.substitutions.append( ('%run_if_sw_emu', run_if_sw_emu) )

# Set timeout for test = 10 mins
try:
    import psutil
    lit_config.maxIndividualTestTime = 10800
except ImportError:
    pass
