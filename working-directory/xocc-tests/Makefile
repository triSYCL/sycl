# Not intended to be a long-term solution, this is a triSYCL style hack to see
# how XOCC handles the output of ISYCL. Better implementation would be to
# integrate a Xilinx SYCL executable bundler into Clang, support exists for
# Intel SYCL at the moment
CXX = $(ISYCL_BIN_DIR)/clang++
OPT = $(ISYCL_BIN_DIR)/opt
LINK =  $(ISYCL_BIN_DIR)/llvm-link

# only define this if you intend not to use the shell script and only aim to
# compile 1 kernel in a file
#XOCC_KERNEL_NAME = kernel_name

# This should be a path to the libspir64-39-hls.bc file, it's definitions for
# all of the spir intrinsics (_Z13get_global_idj etc.) that get linked in by
# XOCC. This is a workaround for a bug(?) in the XOCC compiler that doesn't add
# this library in when using the xpirbc path. Note: Alexandre thinks that it may
# be feasible to use opencl-c.h, but I may be taking that out of context and I've
# yet to test it.
LIB_SPIR=/net/xsjswsvm1-lif1/hdstaff1/isoard/Perforce/Rodin/REL/2018.3/prep/rdi/vivado/lnx64/lib/libspir64-39-hls.bc

# Specify a target device, otherwise it defaults to xilinx_vcu1525_dynamic_5_1
ifdef TARGET_DEVICE
  DEVICE_XOCC := $(TARGET_DEVICE)
else
  DEVICE_XOCC := xilinx_vcu1525_dynamic_5_1
endif

ifdef XCL_EMULATION_MODE
  DEVICE_MODE := $(XCL_EMULATION_MODE)
else
  DEVICE_MODE := hw
endif

XOCC = xocc # currently assuming xocc exists in your path
XOCCFLAGS = --platform $(DEVICE_XOCC) --save-temps

LDLIBS += -lsycl -lOpenCL

print-%  : ; @echo $* = $($*)

# source code device path to spir, I think -fsycl-int-header= defines the name of the integrated header it will generate
# so in theory it can be unrelated to the .cpp file.
# found using -###
# TODO:
# 1: Remove specificness to my directorys, perhaps possible by just removing them and letting clang search for them? Maybe there is a simple way to include isystem somewhere
%-int-header.h %.sp: %.cpp
	$(CXX) -cc1 -O3 -triple spir64-unknown-linux-sycldevice -std=c++11 \
	-fsycl-is-device -aux-triple x86_64-unknown-linux-gnu \
	-emit-llvm -disable-free -disable-llvm-verifier -discard-value-names \
	-main-file-name $* -mrelocation-model \
	static -mthread-model posix -mdisable-fp-elim -fmath-errno -no-integrated-as \
	-mconstructor-aliases -fuse-init-array -dwarf-column-info \
	-debugger-tuning=gdb -coverage-notes-file \
	/storage/ogozillo/intel-sycl/xocc-tests/$*.gcno \
	-resource-dir /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0 \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include/sycl_wrappers \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/x86_64-linux-gnu/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/spir64-unknown-linux-sycldevice/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8/backward -internal-isystem /usr/local/include \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu \
	-internal-externc-isystem /include -internal-externc-isystem /usr/include -fdeprecated-macro -fno-dwarf-directory-asm \
	-fdebug-compilation-dir /storage/ogozillo/intel-sycl/xocc-tests -ferror-limit 19 -fmessage-length 247 \
	-fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -fcolor-diagnostics \
	-fsycl-int-header=$*-int-header.h -o kernel.sp -x c++ $*.cpp

# -fsycl-no-kernel-name-mangle
# -fsycl-int-header-no-mangle (it'll still mangle the llvm ir, so it would need
#															 to demangle inside InSPIRation?)
# -fsycl-header-seq-kernel-name (rename kernels in a file sequentially)
# -fsycl-xocc (a little unclear to a user what will occur in this case and I
#							think specifying xocc(xocl?) is more suitable as a triple,
#							it does allow more things to go on though)


%-int-header.h %.spv: %.cpp
	$(CXX) -cc1 -O3 -triple spir64-unknown-linux-sycldevice -std=c++11 \
	-fsycl-is-device -aux-triple x86_64-unknown-linux-gnu \
	-emit-spirv -disable-free -disable-llvm-verifier -discard-value-names \
	-main-file-name $* -mrelocation-model \
	static -mthread-model posix -mdisable-fp-elim -fmath-errno -no-integrated-as \
	-mconstructor-aliases -fuse-init-array -dwarf-column-info \
	-debugger-tuning=gdb -coverage-notes-file \
	/storage/ogozillo/intel-sycl/xocc-tests/$*.gcno \
	-resource-dir /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0 \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include/sycl_wrappers \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/x86_64-linux-gnu/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/spir64-unknown-linux-sycldevice/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8/backward -internal-isystem /usr/local/include \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu \
	-internal-externc-isystem /include -internal-externc-isystem /usr/include -fdeprecated-macro -fno-dwarf-directory-asm \
	-fdebug-compilation-dir /storage/ogozillo/intel-sycl/xocc-tests -ferror-limit 19 -fmessage-length 247 \
	-fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -fcolor-diagnostics \
	-fsycl-int-header=$*-int-header.h -o kernel.spv -x c++ $*.cpp


%.int-header:
	$(CXX) -cc1 -O3 -triple spir64-unknown-linux-sycldevice -std=c++11 \
	-fsycl-is-device -aux-triple x86_64-unknown-linux-gnu \
	-emit-llvm -disable-free -disable-llvm-verifier -discard-value-names \
	-main-file-name $* -mrelocation-model \
	static -mthread-model posix -mdisable-fp-elim -fmath-errno -no-integrated-as \
	-mconstructor-aliases -fuse-init-array -dwarf-column-info \
	-debugger-tuning=gdb -coverage-notes-file \
	/storage/ogozillo/intel-sycl/xocc-tests/$*.gcno \
	-resource-dir /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0 \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include/sycl_wrappers \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/x86_64-linux-gnu/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/spir64-unknown-linux-sycldevice/c++/8 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/8/../../../../include/c++/8/backward -internal-isystem /usr/local/include \
	-internal-isystem /storage/ogozillo/intel-sycl/build/lib/clang/8.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu \
	-internal-externc-isystem /include -internal-externc-isystem /usr/include -fdeprecated-macro -fno-dwarf-directory-asm \
	-fdebug-compilation-dir /storage/ogozillo/intel-sycl/xocc-tests -ferror-limit 19 -fmessage-length 247 \
	-fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -fcolor-diagnostics \
	-fsycl-int-header=$*-int-header.h -o kernel.sp -x c++ $*.cpp
	rm kernel.sp

# This makefile needs cleaned and refactored.
# 1) Change the .sp to .bin, just easier to have a unified naming convention

#
# 1) Do I lose Qualifiers not mangling the name?
# 2) Do we pollute the .sp type list with ampersands ourselves when running the
# 	inspiration opt pass?

# probably needs to in someway accept the fsycl-int-header, or we need to just assume the name and search
%.bc: %.sp
	# -reqd-workgroup-size-1 currently crashes the runtime when an clEnqueueNDRangeKernel
	#  is executed, crashes with an invalid work group size opencl error
	$(OPT) -inSPIRation -globaldce -o $*.bc kernel.sp

%.sp_opt: %.sp
	$(OPT) -S -inSPIRation -globaldce -o kernel.sp_opt kernel.sp

# The problem at the moment is that XOCC needs to know what the OCL intrinsics
# are as it only has declarations and can't seem to map the definitions to them
%.xo: %.bc
	cp -p $< $*.xpirbc
	# this will complain about wrong data layouts and triples, should be fine
	# though as xocc overwrites these anyway for the desired target. Getting rid
	# of them would require us to hardcore the layout and triple in the opt pass
	# otherwise.
	# TODO: Will a -globaldce pass after linking speed up compilation or is that
	# redundant? Xocc will do it anyway, but is an opt pass removing unneeded
	# library components and then sending it to xocc quicker?..
	$(LINK) -o $*.xpirbc $*.xpirbc $(LIB_SPIR)
	$(XOCC) $(XOCCFLAGS) --target $(DEVICE_MODE) -c \
	  -k $(XOCC_KERNEL_NAME) -o $@ $*.xpirbc
	rm $*.xpirbc

%.bin: %.xo
	$(XOCC) $(XOCCFLAGS) --target $(DEVICE_MODE) -l \
	  -k $(XOCC_KERNEL_NAME) -o kernel.bin $*.xo

%.linked_bin:
	$(XOCC) $(XOCCFLAGS) --target $(DEVICE_MODE) -l \
		 -o kernel.bin $(LINKER_LIST)

%.sycl_exe:
	$(CXX) -std=c++11 -g -include $*-int-header.h $< $*.cpp -o $*.sycl_exe $(LDLIBS)

# The only real point in the below is to force the makefile to compile certain
# things other than that they're the same command
%.sycl_xocc_exe: %-int-header.h %.cpp %.bin
	$(CXX) -std=c++11 -g -include $*-int-header.h $*.cpp -o $*.sycl_xocc_exe $(LDLIBS)

%.sycl_sp_exe: %-int-header.h %.cpp %.sp
	$(CXX) -std=c++11 -g -include $< $*.cpp -o $*.sycl_sp_exe $(LDLIBS)

%.sycl_spv_exe: %-int-header.h %.cpp %.spv
	$(CXX) -std=c++11 -g -include $< $*.cpp -o $*.sycl_spv_exe $(LDLIBS)

# clean:
# 	$(RM)

# Relevant Grep's
# llvm::Triple::SYCLDevice
# getLangOpts().SYCL
# LangOpts.SYCL

#cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DOpenCL_INCLUDE_DIR=$OPENCL_HEADERS -DLLVM_ENABLE_CXX1Y=ON -DLLVM_LINK_LLVM_DYLIB:BOOL=ON -DLLVM_BUILD_LLVM_DYLIB:BOOL=ON -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_EXTERNAL_PROJECTS="sycl;llvm-spirv" -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/sycl -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm-spirv -DLLVM_TOOL_SYCL_BUILD=ON -DLLVM_TOOL_LLVM_SPIRV_BUILD=ON $SYCL_HOME/llvm

# Relevant Files for change:
# /llvm-intel/clang/lib/CodeGen/CodeGenModule.cpp, generates header for SPIRV (https://github.com/intel/llvm/commit/f1a7138c3efae93e1843bbddf7ed0cabbab33c93)
