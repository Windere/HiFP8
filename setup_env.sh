#!/bin/bash
# HiFP8 环境设置脚本

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/ao:${PYTHONPATH}"

echo "=================================="
echo "HiFP8 环境已配置"
echo "=================================="
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""
echo "可用命令:"
echo "  python examples/quantize_model.py              # 运行简单 demo"
echo "  python -m unittest tests.test_hifp8_flow -v    # 运行测试"
echo ""
