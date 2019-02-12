#!/bin/bash
# Simple script that leverages the Makefile to generate an xocc kernel binary (xclbin) that contains more than one xpirbc/xo.
# Naively expects to be in the same folder as the Makefile and the file its intended to compile.  
# Shouldn't be invoked using source/. unless you want your terminals environment polluted with some extra exports... Currently
# using export to pass "parameters" down to the Makefile which should be wiped once the shell script exits. 
# NOTE: Not the most optimal shell and Makefile script combination but it seems to get the job done for now.

usage() { echo xocc_compile: error: $2 >&2; exit $1; }

[ $# -eq 0      ] && usage 1 "no input files"

# no real need for the EXT at the moment, just here in case.
EXT=${1##*.}
NAME=${1%%.*}

# force header compilation 
make $NAME.int-header

# load in header to a shell variable
HEADER=$(<$NAME-int-header.h)

# Regex for extracting kernel name from the headers array of char*:
#static constexpr
#const char* const kernel_names[] = {
#  "par_1d",
#};
RE="kernel_names\[\]\s=\s\{([^}]+)\}"

# match file against regex then cleanup the string by truncating unwanted characters
if [[ $HEADER =~ $RE ]]; then  
    STR="$(echo "${BASH_REMATCH[1]}" | tr '\n' ' ')"
    STR="$(echo "${BASH_REMATCH[1]}" | tr ',' ' ')"
    STR="$(echo "$STR" | tr '"' ' ')"
fi


# Reads in shell variable using IFS, checks if the variable is nullary or not then truncates excess space and exports the variable 
# (using it as a parameter to the Makefile in this case, same as the LINKER_LIST) then compiles the file for the speciifc kernel, 
# renames and adds to the list to be linked (basically invoking xocc -c for all the kernels in a file to generate .xo's individually)
# NOTE: Can probably be optimized invoking the make file with .bin is probably rerunning the intel-sycl compiler and opt phase each time. 
# It is a small amount of the execution time however.
while IFS= read -r X
do
if [[ -n "$X" ]]; then
    X="$(echo -e "${X}" | tr -d '[:space:]')"
    export XOCC_KERNEL_NAME=$X
    make $NAME.xo 
    mv "$NAME.xo" "$X.xo"
    LINKER_LIST="$X.xo $LINKER_LIST"
fi
done <<< $STR 

# export our list of .xo's to be linked
echo "Linking: $LINKER_LIST"
export LINKER_LIST=$LINKER_LIST

# invoke linking of all of our .xos in an xocc -l stage
make $NAME.linked_bin

# generate the host side executeable using the integrated device header
make $NAME.sycl_exe

# Perform cleanups of extra files
echo "Deleting: $LINKER_LIST"
for OBJECT in $LINKER_LIST
do
 rm $OBJECT
done

echo "Deleting intermediate kernel.sp file"
rm kernel.sp

echo "Deleting excess XOCC .log and .info files as well as _x (Vivado project) folder"
rm *.log
rm *.info
rm -rf _x
