#!/bin/bash
# Copyright (c) 2023-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# Terms of the MIT License

workdir="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
BUILD_DIR=$(cd `dirname $0`; pwd)/build
op_type="install"
install_prefix=/usr/local/musa

usage() {
    echo -e "musa toolkits install script"
    echo -e "Usage:"
    echo -e "$0 <options>"
    echo -e "   [-i | --install]"
    echo -e "   [-u | --uninstall]"
    echo -e "   [-d | --prefix=<install_prefix>]"
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

install_func () {
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR}
    cmake \
      -DCMAKE_INSTALL_PREFIX=${install_prefix} \
      -DCUB_ENABLE_HEADER_TESTING=OFF \
      -DCUB_ENABLE_TESTING=OFF \
      -DCUB_ENABLE_EXAMPLES=OFF \
      .. 2>&1 | tee cmake.log
    cmake --build .  2>&1 | tee build.log
    if [ ! -w ${install_prefix} ]; then
      sudo cmake --build . --target install  2>&1 | tee install.log
    else
      cmake --build . --target install  2>&1 | tee install.log
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

while getopts d:-:iuh OPT; do
    if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi
    case "$OPT" in
        i | install )    op_type=install;;
        u | uninstall )    op_type=uninstall;;
        d | prefix )        needs_arg; install_prefix="$OPTARG";;
        h | help )          usage; exit 2;;
        ??* )               die "Illegal option --$OPT" ;;  # bad long option
        ? )                 exit 2 ;;  # bad short option (error reported via getopts)
    esac
done

if [[ "$op_type" == "install" ]]; then
  echo "install  ..."
  install_func
fi

if [[ "$op_type" == "uninstall" ]]; then
  echo "uninstall  ..."
  uninstall_func
fi
