#!/bin/bash
# CUB 并行编译和测试脚本 (使用 Ninja)
#
# 所有参数均可通过环境变量设置，命令行参数优先级更高
# 环境变量:
#   CUB_JOBS          编译并行数 (默认: nproc)
#   CUB_TEST_JOBS     测试并行数 (默认: 1)
#   CUB_MUSA_ARCH     MUSA 目标架构 (默认: mp_31)
#   CUB_MUSA_DEVICES  设置 MUSA_VISIBLE_DEVICES
#   CUB_NO_CLEAN      设置为 1 不删除 build 目录
#   CUB_EXCLUDE_TESTS 排除匹配正则表达式的测试
#   CUB_BUILD_ONLY    设置为 1 仅编译不测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUB_DIR="${SCRIPT_DIR}"
SOURCE_DIR="${CUB_DIR}"

# 默认值 - 可通过环境变量覆盖
JOBS="${CUB_JOBS:-$(nproc)}"
TEST_JOBS="${CUB_TEST_JOBS:-1}"
RUN_TEST=true
TEST_VERBOSE="-V"
MUSA_DEVICES="${CUB_MUSA_DEVICES:-}"  # 默认所有GPU可见
SKIP_CLEAN="${CUB_NO_CLEAN:-false}"
BUILD_ONLY="${CUB_BUILD_ONLY:-false}"
EXCLUDE_TESTS="${CUB_EXCLUDE_TESTS:-grid_barrier|namespace_wrapped}"  # grid_barrier 和 namespace_wrapped 会挂起
MUSA_ARCH="${CUB_MUSA_ARCH:-mp_31}"  # 默认 MUSA 架构
BUILD_DIR=""  # 将在参数解析后设置
LOG_FILE=""
REPORT_FILE=""

# Thrust 相关
MUSA_INCLUDE_DIR="/usr/local/musa/include"
THRUST_REPO="git@sh-code.mthreads.com:sw/muThrust.git"
THRUST_BRANCH="develop-1.17"

show_help() {
    cat << EOF
用法: $0 [选项] [命令]

命令:
  clean         仅清理 build 目录
  build         仅编译，不运行测试

选项:
  -j, --jobs N      编译并行数 (默认: $(nproc))
  -T, --test-jobs N 测试并行数 (默认: 8)
  -g, --gpus DEVICES 设置 MUSA_VISIBLE_DEVICES (如: 0,1,2,3)
  -a, --arch ARCH   MUSA 目标架构 (默认: mp_31, 支持: mp_21, mp_22, mp_31)
  -b, --build-dir DIR 指定构建目录名 (默认: build_<arch>, 如 build_mp_31)
  -n, --no-clean    不删除 build 目录 (增量编译)
  -E, --exclude RE  排除匹配正则表达式的测试 (默认: ${EXCLUDE_TESTS})
                    传空字符串 "" 可取消默认排除
  -h, --help        显示帮助

默认行为:
  1. 检查并安装 thrust (如果缺失)
  2. 删除 build 目录
  3. CMake 配置
  4. 编译
  5. 运行测试 (ctest -V)
  6. 生成 markdown 报告

示例:
  $0                          # 完整流程：清理、编译、测试、生成报告 (使用 build_mp_31)
  $0 -a mp_22                 # 使用 build_mp_22 目录
  $0 build                    # 仅编译
  $0 -n                       # 增量编译并测试
  $0 -T 4 -g 0,1,2,3          # 用4个并行测试，只用GPU 0-3
  $0 --build-dir custom       # 使用自定义构建目录 build_custom
EOF
}

check_and_install_thrust() {
    if [ -d "${MUSA_INCLUDE_DIR}/thrust" ]; then
        echo "Thrust 已安装: ${MUSA_INCLUDE_DIR}/thrust"
        return 0
    fi

    echo "检测到 Thrust 未安装，正在从 ${THRUST_REPO} 下载..."

    TEMP_DIR=$(mktemp -d)
    trap "rm -rf ${TEMP_DIR}" EXIT

    cd "${TEMP_DIR}"
    git clone --depth 1 --branch "${THRUST_BRANCH}" "${THRUST_REPO}" muThrust

    echo "正在安装 Thrust 到 ${MUSA_INCLUDE_DIR}..."
    sudo cp -r muThrust/thrust "${MUSA_INCLUDE_DIR}/"

    echo "Thrust 安装完成"
    cd "${CUB_DIR}"
}

# 解析参数
CUSTOM_BUILD_DIR=""
DO_CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -T|--test-jobs)
            TEST_JOBS="$2"
            shift 2
            ;;
        -g|--gpus)
            MUSA_DEVICES="$2"
            shift 2
            ;;
        -a|--arch)
            MUSA_ARCH="$2"
            shift 2
            ;;
        -b|--build-dir)
            CUSTOM_BUILD_DIR="$2"
            shift 2
            ;;
        -E|--exclude)
            EXCLUDE_TESTS="$2"
            shift 2
            ;;
        -n|--no-clean)
            SKIP_CLEAN=true
            shift
            ;;
        clean)
            DO_CLEAN=true
            shift
            ;;
        build)
            BUILD_ONLY=true
            RUN_TEST=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置构建目录 (基于架构自动命名，除非指定了自定义目录)
if [ -n "$CUSTOM_BUILD_DIR" ]; then
    BUILD_DIR="${CUB_DIR}/build_${CUSTOM_BUILD_DIR}"
else
    BUILD_DIR="${CUB_DIR}/build_${MUSA_ARCH}"
fi
LOG_FILE="${BUILD_DIR}/test_verbose.log"
REPORT_FILE="${CUB_DIR}/test_report_${MUSA_ARCH}.md"

# 处理 clean 命令 (需要在设置 BUILD_DIR 后)
if [ "$DO_CLEAN" = true ]; then
    echo "清理构建目录: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# 1. 检查并安装 thrust
check_and_install_thrust

cd "${CUB_DIR}"

# 2. 清理 build 目录
if [ "$SKIP_CLEAN" = false ] && [ -d "${BUILD_DIR}" ]; then
    echo "删除 build 目录..."
    rm -rf "${BUILD_DIR}"
fi

# 3. CMake 配置
echo ""
echo "=========================================="
echo "CMake 配置 (Ninja)..."
echo "MUSA 架构: ${MUSA_ARCH}"
echo "=========================================="
cmake -G Ninja \
    -DMUSA_64_BIT_DEVICE_CODE=ON \
    -DMUSA_ARCH_LIST="${MUSA_ARCH}" \
    -DCUB_ENABLE_TESTING=ON \
    -DCUB_ENABLE_EXAMPLES=ON \
    -DCUB_ENABLE_HEADER_TESTING=OFF \
    -S "${SOURCE_DIR}" -B "${BUILD_DIR}"

# 4. 并行编译
echo ""
echo "=========================================="
echo "编译 (并行数: ${JOBS})..."
echo "=========================================="
cmake --build "${BUILD_DIR}" -j "${JOBS}"

if [ "$RUN_TEST" = true ]; then
    echo ""
    echo "=========================================="
    echo "运行测试 (并行数: ${TEST_JOBS})"
    echo "=========================================="

    # 设置环境变量
    if [ -n "$MUSA_DEVICES" ]; then
        export MUSA_VISIBLE_DEVICES="$MUSA_DEVICES"
        echo "MUSA_VISIBLE_DEVICES=${MUSA_DEVICES}"
    fi

    # 运行 ctest 并保存输出
    echo "测试输出保存到: ${LOG_FILE}"
    EXCLUDE_ARG=""
    if [ -n "$EXCLUDE_TESTS" ]; then
        EXCLUDE_ARG="-E ${EXCLUDE_TESTS}"
        echo "排除测试: ${EXCLUDE_TESTS}"
    fi
    ctest --test-dir "${BUILD_DIR}" -j "${TEST_JOBS}" ${TEST_VERBOSE} ${EXCLUDE_ARG} 2>&1 | tee "${LOG_FILE}"

    # 生成 markdown 报告
    echo ""
    echo "=========================================="
    echo "生成测试报告"
    echo "=========================================="
    python3 "${CUB_DIR}/parse_ctest_log.py" "${LOG_FILE}" "${REPORT_FILE}"
    echo "测试报告: ${REPORT_FILE}"
fi

echo ""
echo "完成!"
