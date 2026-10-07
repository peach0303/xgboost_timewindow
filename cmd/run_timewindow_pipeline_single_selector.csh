#!/bin/csh -f
#===============================================================================
# 旧版兼容入口：保留 Step 2 单选择器与“代理信号 × 全部时间窗”特征方式。
#
#   [Step 1] -> [Step 2: PROXY_SELECTOR=mcp|xgb（单选）]
#            -> [Step 3/4: 选代理信号并展开所有可用时间窗]
#            -> [Step 5/6: Two-phase Method A（旧版默认，可覆盖关闭）]
#
# 可覆盖项：
#   PROXY_SELECTOR=mcp|xgb
#   FEATURE_INPUT_MODE=signal_all_windows|proxy_columns
#   TWO_PHASE_METHOD=A|""
#   STEP2_PARALLEL_DSUB=0|1
#
# Usage:
#   csh run_timewindow_pipeline_single_selector.csh <hardware> [all|prep|train_k] [K]
#===============================================================================

set SCRIPT_DIR = `dirname "$0"`

# 此入口的定义就是 Step 2 单选择器；其他旧版默认值允许调用方覆盖。
setenv STEP2_SELECTOR_MODE single
if ( ! $?PROXY_SELECTOR ) setenv PROXY_SELECTOR mcp
if ( ! $?FEATURE_INPUT_MODE ) setenv FEATURE_INPUT_MODE signal_all_windows
if ( ! $?TWO_PHASE_METHOD ) setenv TWO_PHASE_METHOD A

echo "Legacy single-selector pipeline:"
echo "  PROXY_SELECTOR     = ${PROXY_SELECTOR}"
echo "  FEATURE_INPUT_MODE = ${FEATURE_INPUT_MODE}"
if ( "$TWO_PHASE_METHOD" == "" ) then
    echo "  TWO_PHASE_METHOD   = disabled"
else
    echo "  TWO_PHASE_METHOD   = ${TWO_PHASE_METHOD}"
endif

exec csh "${SCRIPT_DIR}/run_timewindow_pipeline.csh" $argv:q
