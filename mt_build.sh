#!/bin/bash
# Copyright @2020-2023 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
#
# This software ("this software and its documentations" or "the software") is
# protected by Copyright and the information contained herein is confidential.
#
# The software contained herein is PROPRIETARY to Moore Threads and is being
# provided under the terms and conditions of a form of Moore Threads software
# license agreement by and between Moore Threads and Licensee ("License
# Agreement") or electronically accepted by Licensee. Notwithstanding any
# terms or conditions to the contrary in the License Agreement, copy or
# disclosure of the software to any third party without the express written
# consent of Moore Threads is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
# AGREEMENT, MOORE THREDS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
# INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
# PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
# ANY KIND. MOORE THREDS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
# SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREDS BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THE SOFTWARE.

workdir="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
BUILD_DIR=$(cd `dirname $0`; pwd)/build
op_type="package"

### the following line is no need to modify, it will be modified automatically by build.sh
install_prefix=/usr/local/musa
package_arch=""

usage() {
    echo -e "musa toolkits install script"
    echo -e "Usage:"
    echo -e "$0 <options>"
    echo -e "   [-p | --package]"
    echo -e "   [-i | --install]"
    echo -e "   [-u | --uninstall]"
    echo -e "   [-d | --prefix=<install_prefix>]"
    echo -e "   [--package_arch=arm64]"
    echo -e "   [-h | --help]"
    echo -e ""
}

die() {
    echo "$*" >&2;
    exit 2;
}

needs_arg() {
    if [ -z "$OPTARG" ]; then
        usage
        die "No arg for --$OPT option";
    fi;
}

element_in () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 1; done
    return 0
}

package_func () {
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR}
    cmake ${cmake_opts} \
      -DCMAKE_INSTALL_PREFIX=${install_prefix} \
      -DCUB_ENABLE_HEADER_TESTING=OFF \
      -DCUB_ENABLE_TESTING=OFF \
      -DCUB_ENABLE_EXAMPLES=OFF \
      .. 2>&1 | tee cmake.log
    cmake --build . -j -v 2>&1 | tee build.log
    cmake --build . -j -v --target package  2>&1 | tee package.log
    popd
}

install_func () {
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR}
    cmake \
      -DCMAKE_INSTALL_PREFIX=${install_prefix} \
      -DCUB_ENABLE_HEADER_TESTING=OFF \
      -DCUB_ENABLE_TESTING=OFF \
      -DCUB_ENABLE_EXAMPLES=OFF \
      .. 2>&1 | tee cmake.log
    cmake --build . -j 2>&1 | tee build.log
    if [ ! -w ${install_prefix} ]; then
      sudo cmake --build . -j --target install  2>&1 | tee install.log
    else
      cmake --build . -j --target install  2>&1 | tee install.log
    fi
    popd
}

uninstall_func () {
    if [ "$UID" -ne "0" ]; then
      sudo \rm -vrf ${install_prefix}/include/cub
      sudo \rm -vrf ${install_prefix}/lib/cmake/cub
      sudo \rm -vf ${install_prefix}/bin/CUB_version
    else
      \rm -vrf ${install_prefix}/include/cub
      \rm -vrf ${install_prefix}/lib/cmake/cub
      \rm -vf ${install_prefix}/bin/CUB_version
    fi
}

while getopts d:-:piuh OPT; do
    if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi
    case "$OPT" in
        p | package )    op_type=package;;
        i | install )    op_type=install;;
        u | uninstall )    op_type=uninstall;;
        d | prefix )        needs_arg; install_prefix="$OPTARG";;
        package_arch )         needs_arg; package_arch="$OPTARG";;
        h | help )          usage; exit 2;;
        ??* )               die "Illegal option --$OPT" ;;  # bad long option
        ? )                 exit 2 ;;  # bad short option (error reported via getopts)
    esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list
OTHER_ARGS=$@
OTHER_ARGS_TMP=( ${OTHER_ARGS} )
OTHER_ARGS_LEN=${#OTHER_ARGS_TMP[@]}
# echo "OTHER_ARGS: ${OTHER_ARGS}"
if [[ ${package_arch} == "arm64" ]]; then
  cmake_opts="${cmake_opts} -DPACKAGE_ARCH_ARM64=ON"
else
  cmake_opts="${cmake_opts} -DPACKAGE_ARCH_ARM64=OFF"
fi

if [[ "$op_type" == "package" ]]; then
  echo "package  ..."
  package_func
fi

if [[ "$op_type" == "install" ]]; then
  echo "install  ..."
  install_func
fi

if [[ "$op_type" == "uninstall" ]]; then
  echo "uninstall  ..."
  uninstall_func
fi
