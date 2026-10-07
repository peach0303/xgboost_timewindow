#!/bin/csh -f
# Compatibility wrapper. The main pipeline now always runs MCP and XGBoost
# proxy selection together; this wrapper only supplies XGBoost tuning defaults.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`

if ( ! $?XGB_PROXY_TOP_K ) setenv XGB_PROXY_TOP_K 200
if ( ! $?XGB_PROXY_ROUNDS ) setenv XGB_PROXY_ROUNDS 50
if ( ! $?XGB_PROXY_MAX_DEPTH ) setenv XGB_PROXY_MAX_DEPTH 4
if ( ! $?XGB_PROXY_ETA ) setenv XGB_PROXY_ETA 0.08
if ( ! $?XGB_PROXY_SUBSAMPLE ) setenv XGB_PROXY_SUBSAMPLE 0.9
if ( ! $?XGB_PROXY_COLSAMPLE ) setenv XGB_PROXY_COLSAMPLE 0.8
if ( ! $?XGB_PROXY_NTHREAD ) setenv XGB_PROXY_NTHREAD 1

csh "${SCRIPT_DIR}/run_timewindow_pipeline.csh" $argv:q
