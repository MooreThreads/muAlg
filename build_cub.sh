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
RUN_TEST=false
TEST_VERBOSE="-V"
MUSA_DEVICES=""  # 默认所有GPU可见
LOG_FILE=""      # 保存输出到文件

show_help() {
    cat << EOF
用法: $0 [选项] [命令]

命令:
  clean         清理 build 目录

选项:
  -j, --jobs N      编译并行数 (默认: $(nproc))
  -t, --test        编译后运行 ctest
  -T, --test-jobs N 测试并行数 (默认: 8)
  -q, --quiet       测试时减少输出 (只显示失败)
  -g, --gpus DEVICES 设置 MUSA_VISIBLE_DEVICES (如: 0,1,2,3)
  -l, --log FILE    保存测试输出到文件 (如: -l test.log)
  -h, --help        显示帮助

示例:
  $0                          # 仅编译
  $0 -t                       # 编译并运行测试
  $0 -t -T 4                  # 编译并用4个并行运行测试
  $0 -t -g 0,1,2,3            # 编译并测试，只用GPU 0-3
  $0 -t -q                    # 编译并测试，减少输出
  $0 clean                    # 清理
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -t|--test)
            RUN_TEST=true
            shift
            ;;
        -T|--test-jobs)
            TEST_JOBS="$2"
            shift 2
            ;;
        -q|--quiet)
            TEST_VERBOSE="--output-on-failure"
            shift
            ;;
        -g|--gpus)
            MUSA_DEVICES="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        clean)
            echo "清理 build 目录..."
            rm -rf "${BUILD_DIR}"
            exit 0
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

cd "${CUB_DIR}"

# CMake 配置
echo "CMake 配置 (Ninja)..."
cmake -G Ninja -DMUSA_64_BIT_DEVICE_CODE=ON -S "${SOURCE_DIR}" -B build

# 并行编译
echo "编译 (并行数: ${JOBS})..."
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

    # 运行 ctest
    if [ -n "$LOG_FILE" ]; then
        echo "测试输出同时保存到: ${LOG_FILE}"
        ctest --test-dir build -j "${TEST_JOBS}" ${TEST_VERBOSE} 2>&1 | tee "${LOG_FILE}"
    else
        ctest --test-dir build -j "${TEST_JOBS}" ${TEST_VERBOSE}
    fi
fi

echo ""
echo "完成!"
