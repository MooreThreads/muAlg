#!/bin/bash
# Copyright (c) 2023-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# Terms of the MIT License

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1; pwd -P)"
BUILD_DIR="${SCRIPT_DIR}/build"
OP_TYPE="install"
INSTALL_PREFIX="/usr/local/musa"

usage() {
    cat <<EOF_USAGE
musa toolkits install script
Usage:
$0 <options>
   [-i | --install]
   [-u | --uninstall]
   [-d | --prefix=<install_prefix>]
   [-h | --help]
EOF_USAGE
}

die() {
    echo "$*" >&2
    exit 2
}

needs_arg() {
    if [ -z "${OPTARG:-}" ]; then
        usage
        die "No arg for --$OPT option"
    fi
}

run_install() {
    mkdir -p "${BUILD_DIR}"
    pushd "${BUILD_DIR}" >/dev/null
    cmake \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DCUB_ENABLE_HEADER_TESTING=OFF \
        -DCUB_ENABLE_TESTING=OFF \
        -DCUB_ENABLE_EXAMPLES=OFF \
        .. 2>&1 | tee cmake.log
    cmake --build . 2>&1 | tee build.log
    if [ ! -w "${INSTALL_PREFIX}" ]; then
        sudo cmake --build . --target install 2>&1 | tee install.log
    else
        cmake --build . --target install 2>&1 | tee install.log
    fi
    popd >/dev/null
}

run_uninstall() {
    if [ "$UID" -ne "0" ]; then
        sudo rm -vrf "${INSTALL_PREFIX}/include/cub"
        sudo rm -vrf "${INSTALL_PREFIX}/lib/cmake/cub"
        sudo rm -vrf "${INSTALL_PREFIX}/lib64/cmake/cub"
        sudo rm -vf "${INSTALL_PREFIX}/bin/CUB_version"
    else
        rm -vrf "${INSTALL_PREFIX}/include/cub"
        rm -vrf "${INSTALL_PREFIX}/lib/cmake/cub"
        rm -vrf "${INSTALL_PREFIX}/lib64/cmake/cub"
        rm -vf "${INSTALL_PREFIX}/bin/CUB_version"
    fi
}

while getopts d:-:iuh OPT; do
    if [ "$OPT" = "-" ]; then
        OPT="${OPTARG%%=*}"
        OPTARG="${OPTARG#$OPT}"
        OPTARG="${OPTARG#=}"
    fi
    case "$OPT" in
        i | install ) OP_TYPE="install" ;;
        u | uninstall ) OP_TYPE="uninstall" ;;
        d | prefix ) needs_arg; INSTALL_PREFIX="$OPTARG" ;;
        h | help ) usage; exit 0 ;;
        ??* ) die "Illegal option --$OPT" ;;
        ? ) exit 2 ;;
    esac
done

if [ "${OP_TYPE}" = "install" ]; then
    echo "install ..."
    run_install
fi

if [ "${OP_TYPE}" = "uninstall" ]; then
    echo "uninstall ..."
    run_uninstall
fi
