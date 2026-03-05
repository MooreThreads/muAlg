#!/bin/bash
# CUB 并行编译和测试脚本 (使用 Ninja)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUB_DIR="${SCRIPT_DIR}"
BUILD_DIR="${CUB_DIR}/build"
SOURCE_DIR="${CUB_DIR}"

# 默认值
JOBS=$(nproc)
TEST_JOBS=8
RUN_TEST=true
TEST_VERBOSE="-V"
MUSA_DEVICES=""  # 默认所有GPU可见
LOG_FILE="${CUB_DIR}/test_verbose.log"
REPORT_FILE="${CUB_DIR}/test_report.md"
SKIP_CLEAN=false
BUILD_ONLY=false

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
  -n, --no-clean    不删除 build 目录 (增量编译)
  -h, --help        显示帮助

默认行为:
  1. 检查并安装 thrust (如果缺失)
  2. 删除 build 目录
  3. CMake 配置
  4. 编译
  5. 运行测试 (ctest -V)
  6. 生成 markdown 报告

示例:
  $0                          # 完整流程：清理、编译、测试、生成报告
  $0 build                    # 仅编译
  $0 -n                       # 增量编译并测试
  $0 -T 4 -g 0,1,2,3          # 用4个并行测试，只用GPU 0-3
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
        -n|--no-clean)
            SKIP_CLEAN=true
            shift
            ;;
        clean)
            echo "清理 build 目录..."
            rm -rf "${BUILD_DIR}"
            exit 0
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
echo "=========================================="
cmake -G Ninja -DMUSA_64_BIT_DEVICE_CODE=ON -S "${SOURCE_DIR}" -B build

# 4. 并行编译
echo ""
echo "=========================================="
echo "编译 (并行数: ${JOBS})..."
echo "=========================================="
cmake --build build -j "${JOBS}"

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
    ctest --test-dir build -j "${TEST_JOBS}" ${TEST_VERBOSE} 2>&1 | tee "${LOG_FILE}"

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
