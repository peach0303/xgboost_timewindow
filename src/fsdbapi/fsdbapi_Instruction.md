# FSDBAPI 说明

## 1. 文档目的

本文档说明 `/home/yu.zhang/workspace/fsdbapi` 工程的内容、依赖环境、Verdi/FsdbReader 版本、内部 API 调用路径、不同操作目录下需要修改的路径参数，以及从 FSDB 转换到降维、MCP 实验的标准使用步骤。

`fsdbapi` 的目标是绕过传统：

```text
FSDB -> fsdbreport -> bit CSV -> Python 读 CSV -> TOG cache
```

改为：

```text
FSDB -> FsdbReader API -> bit-level TOG sparse cache -> 降维脚本直接读 cache manifest
```

这样可以显著减少 CSV 生成和 Python 解析大 CSV 的时间与 I/O 开销。

## 2. 工程文件说明

工程根目录：

```text
/home/yu.zhang/workspace/fsdbapi
```

主要文件：

```text
Makefile
README.md
src/fsdb_to_tog_cache.cpp
bin/fsdb_to_tog_cache.py
bin/merge_tog_cache_manifests.py
bin/run_boom_coremark1_fsdbapi.csh
bin/run_xiangshan_fsdbapi.csh
bin/run_xiangshan_2fsdbapi.csh
```

主要功能：

| 文件 | 说明 |
|---|---|
| `src/fsdb_to_tog_cache.cpp` | C++ 核心导出器，直接调用 Verdi FsdbReader API 读取 FSDB |
| `bin/fsdb_to_tog_cache.py` | Python wrapper，负责自动 build 并调用 C++ 导出器 |
| `bin/merge_tog_cache_manifests.py` | 合并多个窗口/多个 FSDB 的 TOG cache manifest |
| `bin/run_xiangshan_fsdbapi.csh` | 单个香山 FSDB 的 2000ns 窗口转换脚本 |
| `bin/run_xiangshan_2fsdbapi.csh` | 多个香山 FSDB、共 20 个 2000ns 窗口转换脚本 |
| `Makefile` | 编译 C++ 导出器，配置 Verdi/FsdbReader include/lib 路径 |

生成的核心输出：

```text
<output>/_source_exports/<window>/_work/tog_cache_manifest.json
<output>/_source_exports/<window>/_work/tog_cache/*.tog.cache.csc.npz
<output>/merged/_work/tog_cache_manifest.json
<output>/merged/_work/tog_cache/*.tog.cache.csc.npz
```

其中 `merged/_work/tog_cache_manifest.json` 是后续降维脚本直接读取的入口。

## 3. Verdi/FsdbReader 版本与编译依赖

当前工程默认 Verdi 版本：

```text
Verdi R-2020.12-SP2
```

默认安装路径：

```text
/home/yu.zhang/workspace/tools/verdi/verdi/R-2020.12-SP2
```

Makefile 中的关键配置：

```makefile
VERDI_HOME ?= /home/yu.zhang/workspace/tools/verdi/verdi/R-2020.12-SP2
FSDB_READER_DIR ?= $(VERDI_HOME)/share/FsdbReader
FSDB_READER_PLATFORM ?= linux64
```

include 路径：

```text
$(VERDI_HOME)/share/FsdbReader
```

### 3.1 Verdi 路径变化时如何修改

如果 Verdi 安装目录变化，不建议直接改源码，优先在 make 时覆盖 `VERDI_HOME`：

```bash
cd /home/yu.zhang/workspace/fsdbapi
make clean
make VERDI_HOME=/new/path/to/verdi/R-2020.12-SP2
```

如果 Verdi 版本变化，例如切换到新的 Verdi 安装，需要同时确认：

```text
/new/path/to/verdi/<version>/share/FsdbReader/ffrAPI.h
/new/path/to/verdi/<version>/share/FsdbReader/linux64/libnffr.so
/new/path/to/verdi/<version>/share/FsdbReader/linux64/libnsys.so
```

由于当前 Makefile 使用了：

```makefile
-Wl,-rpath,$(FSDB_LIB_DIR)
```

运行时通常不需要额外设置 `LD_LIBRARY_PATH`。如果二进制被移动到其他机器或 Verdi 路径变化，需要重新编译，或者临时设置：

```bash
export LD_LIBRARY_PATH=/new/path/to/verdi/R-2020.12-SP2/share/FsdbReader/linux64:${LD_LIBRARY_PATH}
```

## 4. FsdbReader API 内部调用路径

C++ 核心文件：

```text
/home/yu.zhang/workspace/fsdbapi/src/fsdb_to_tog_cache.cpp
```

头文件：

```cpp
#include "ffrAPI.h"
```

内部 API 调用流程如下：

### 4.1 检查并打开 FSDB

```cpp
ffrObject::ffrIsFSDB(...)
ffrObject::ffrOpen3(...)
```

用途：

1. 判断输入文件是否为 FSDB。
2. 打开 FSDB，获得 `ffrObject *fsdb_obj`。

### 4.2 读取 scope/variable 树

```cpp
fsdb_obj->ffrGetScopeSeparator()
fsdb_obj->ffrSetTreeCBFunc(tree_cb, &ctx)
fsdb_obj->ffrReadScopeVarTree()
```

用途：

1. 获取层级分隔符。
2. 注册 tree callback。
3. 遍历 FSDB 里的 scope 和 variable。
4. 收集支持的信号。

当前支持的信号要求：

```text
bytes_per_bit == FSDB_BYTES_PER_BIT_1B
```

支持的变量类型包括：

```text
FSDB_VT_VCD_REG
FSDB_VT_VCD_WIRE
FSDB_VT_VCD_TRI
FSDB_VT_VCD_TRIAND
FSDB_VT_VCD_TRIOR
FSDB_VT_VCD_TRIREG
FSDB_VT_VCD_TRI0
FSDB_VT_VCD_TRI1
FSDB_VT_VCD_WAND
FSDB_VT_VCD_WOR
FSDB_VT_VCD_INTEGER
FSDB_VT_VCD_PARAMETER
FSDB_VT_VHDL_SIGNAL
FSDB_VT_VHDL_VARIABLE
FSDB_VT_VHDL_CONSTANT
```

会跳过：

```text
dummy var
void var
unexpanded memory var
非 1-byte-per-bit 编码信号
real/memory 等不适合 bit-level TOG 的数据
```

### 4.3 设定读取时间窗口

```cpp
fsdb_obj->ffrSetViewWindow(&vw_start, &vw_end)
```

用途：

限制 FsdbReader 在指定时间范围内遍历 value change。

### 4.4 按 block 加载信号

```cpp
fsdb_obj->ffrAddToSignalList(idcode)
fsdb_obj->ffrLoadSignals()
```

用途：

按照 `RAW_COL_BLOCK` 控制的信号 block 分批加载，避免一次性加载全设计导致内存过高。

### 4.5 初始化采样起点状态

```cpp
ffrVCTrvsHdl h = fsdb_obj->ffrCreateVCTrvsHdl(idcode)
h->ffrHasIncoreVC()
h->ffrGotoXTag(&q)
h->ffrGetVC(&vc_ptr)
h->ffrFree()
```

用途：

在窗口起点 `start` 前后获取当前电平状态，作为后续 TOG XOR 的初始状态。

### 4.6 按时间遍历 value change

```cpp
ffrTimeBasedVCTrvsHdl tb =
  fsdb_obj->ffrCreateTimeBasedVCTrvsHdl(sig_count, sig_arr.data())

tb->ffrGetVarIdcodeXTagVCSeqNum(&event_id, &event_time, &event_vc, &event_seq)
tb->ffrGotoNextVC()
tb->ffrFree()
```

用途：

按时间顺序遍历当前 block 中所有信号的 value change，并在固定采样点生成 bit-level TOG。

### 4.7 卸载信号并关闭 FSDB

```cpp
fsdb_obj->ffrUnloadSignals()
fsdb_obj->ffrResetSignalList()
fsdb_obj->ffrClose()
```

用途：

释放当前 block 加载的信号数据，为下一个 block 或下一个 FSDB 做准备。

## 5. TOG 生成策略

默认模式：

```bash
--tog-mode sample-xor
```

含义：

```text
TOG[row] = level[row] XOR level[row - 1]
```

另一个模式：

```bash
--tog-mode any-change
```

含义：

```text
如果一个 bit 在相邻采样区间内变化过至少一次，则该 row 记为 1。
```

香山功耗标签对齐推荐使用：

```bash
--tog-mode sample-xor
--end-inclusive
--drop-first-tog-row
```

例如：

```text
start = 20000
end   = 22000
step  = 2
```

采样电平点：

```text
20000, 20002, 20004, ..., 22000
```

共 1001 个 level sample。

经过 XOR 后丢弃第一行 dummy TOG：

```text
20000->20002
20002->20004
...
21998->22000
```

最终输出 1000 行 TOG，与每 2000ns、2ns 周期的功耗标签窗口对齐。

## 6. X/Z/NC/NV 处理策略

当前策略目标是对齐：

```bash
fsdbreport -csv -of b -bit
```

对于 FsdbReader 返回的 bit 编码：

| 编码 | 处理 |
|---|---|
| `FSDB_BT_VCD_0` | 当前 bit 置 0 |
| `FSDB_BT_VCD_1` | 当前 bit 置 1 |
| `FSDB_BT_VCD_NC` | No Change，保持前值，不标 dirty |
| `FSDB_BT_VCD_NV` | No Value，保持当前/default-zero，不标 dirty |
| `FSDB_BT_VCD_X` | 真 X，采样点仍为 X 时标 dirty |
| `FSDB_BT_VCD_Z` | 真 Z，采样点仍为 Z 时标 dirty |
| 其他编码 | 记录统计，不直接作为 X/Z 删除依据 |

重要说明：

```text
NC/NV 是 FsdbReader 内部状态，不等价于真实 X/Z。
```

因此当前实现不会因为 NC/NV 误删列。只有真实 `X/Z` 在采样点仍然可见时，该 bit 才会进入 `removed_xz_names`。

多窗口或多 FSDB 合并时，`merge_tog_cache_manifests.py` 会对所有输入 manifest 的 dirty bit 做 union，并重写每个 block cache，使所有窗口使用一致的列集合和顺序。

## 7. 标准使用步骤

### 7.1 编译

```bash
cd /home/yu.zhang/workspace/fsdbapi
make
```

强制重编译：

```bash
make clean
make
```

指定 Verdi 路径：

```bash
make VERDI_HOME=/home/yu.zhang/workspace/tools/verdi/verdi/R-2020.12-SP2
```

### 7.2 单窗口转换

```bash
cd /home/yu.zhang/workspace/fsdbapi

./bin/fsdb_to_tog_cache.py \
  --fsdb /path/to/input.fsdb \
  --output-dir /path/to/output/window_20000_22000 \
  --dataset-name coremark_1core_20000_22000 \
  --start 20000 \
  --end 22000 \
  --step 2 \
  --raw-col-block 100000 \
  --tog-mode sample-xor \
  --end-inclusive \
  --drop-first-tog-row \
  --verbose
```

输出：

```text
/path/to/output/window_20000_22000/_work/tog_cache_manifest.json
/path/to/output/window_20000_22000/_work/tog_cache/*.tog.cache.csc.npz
/path/to/output/window_20000_22000/original_headers.txt
```

### 7.3 香山两个 FSDB、20 个窗口转换

默认脚本：

```text
/home/yu.zhang/workspace/fsdbapi/bin/run_xiangshan_2fsdbapi.csh
```

默认输入：

```text
/home/yu.zhang/workspace/powermodeling-demo/step2-hardware-emulation/xiangshan_wave_ns_fsdb/coremark_1core_20000_40000_1ns.fsdb
/home/yu.zhang/workspace/powermodeling-demo/step2-hardware-emulation/xiangshan_wave_ns_fsdb/coremark_1core_40000_60000_1ns.fsdb
```

默认输出：

```text
/home/yu.zhang/workspace/fsdbapi/out/xiangshan_coremark_1core_20000_60000_bit
```

运行：

```bash
cd /home/yu.zhang/workspace/fsdbapi
./bin/run_xiangshan_2fsdbapi.csh
```

强制重新 build：

```bash
REBUILD=1 ./bin/run_xiangshan_2fsdbapi.csh
```

默认参数：

```bash
WINDOW_NS=2000
STEP=2
INCLUDE_END_SAMPLE=1 #对齐功耗label处理
DROP_FIRST_TOG_ROW=1
RAW_COL_BLOCK=100000
TOG_MODE=sample-xor
MERGE_MANIFESTS=1
```

最终合并 manifest：

```text
/home/yu.zhang/workspace/fsdbapi/out/xiangshan_coremark_1core_20000_60000_bit/merged/_work/tog_cache_manifest.json
```

该 `manifest.json` 是后续降维入口。

## 8. 不同操作目录下需要修改的路径

### 8.1 fsdbapi 工程目录变化

如果工程从：

```text
/home/yu.zhang/workspace/fsdbapi
```

移动到：

```text
/new/workspace/fsdbapi
```

需要修改或通过环境变量覆盖：

```bash
FSDBAPI=/new/workspace/fsdbapi
```

适用脚本：

```text
bin/run_xiangshan_fsdbapi.csh
bin/run_xiangshan_2fsdbapi.csh
bin/xiangshanreadbit.csh
```

例如：

```bash
FSDBAPI=/new/workspace/fsdbapi \
OUT=/new/output/xiangshan_coremark_1core_20000_60000_bit \
/new/workspace/fsdbapi/bin/run_xiangshan_2fsdbapi.csh
```

`bin/fsdb_to_tog_cache.py` 内部使用：

```python
ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "fsdb_to_tog_cache"
```

因此直接调用 wrapper 时，通常不需要手工改 `ROOT`，只要脚本相对工程目录结构不变即可。

### 8.2 Verdi 安装路径变化

修改方式：

```bash
make clean
make VERDI_HOME=/new/verdi/path/R-2020.12-SP2
```

或直接修改 Makefile：

```makefile
VERDI_HOME ?= /new/verdi/path/R-2020.12-SP2
```

注意：因为二进制带有 `rpath`，Verdi/FsdbReader 路径变化后建议重新编译。

### 8.3 FSDB 数据目录变化

对于双 FSDB 脚本，覆盖：

```bash
FSDB_0=/new/path/coremark_1core_20000_40000_1ns.fsdb
FSDB_1=/new/path/coremark_1core_40000_60000_1ns.fsdb
FSDB0_START=20000
FSDB0_END=40000
FSDB1_START=40000
FSDB1_END=60000
```

示例：

```bash
FSDB_0=/data/new/coremark_1core_20000_40000_1ns.fsdb \
FSDB_1=/data/new/coremark_1core_40000_60000_1ns.fsdb \
OUT=/data/out/xiangshan_fsdbapi \
./bin/run_xiangshan_2fsdbapi.csh
```
若fsdb文件数继续增多，同理可继续增加文件数

## 9. 输出文件格式说明

### 9.1 cache NPZ

fsdbapi 输出的是 SciPy 兼容的 CSC sparse NPZ：

```text
*.tog.cache.csc.npz
```

内部包含：

```text
data.npy     uint8，全 1
indices.npy  int32 或 int64，行索引
indptr.npy   int64，CSC column pointer
shape.npy    int64[2]
format.npy   b"csc"
```

这些文件可被：

```python
scipy.sparse.load_npz(...)
```

直接读取。

### 9.2 manifest

核心 manifest：

```text
_work/tog_cache_manifest.json
```

关键字段：

```text
cache_format
csv_paths
blocks[].names
blocks[].files
blocks[].n_rows
blocks[].nnz
blocks[].removed_xz_names
xz_removed_count
xz_policy
```

这里的 `csv_paths` 在 fsdbapi 场景中不是实际 CSV，而是 source marker，用于保持和原降维脚本的输入文件抽象一致。

### 9.3 降维 final NPZ

降维脚本保存：

```text
*.final.tog.csr.npz
final.cols.txt
final.namefile.txt
meta.json
```

final NPZ 文件名来自 manifest 中 `csv_paths` 的 stem，例如：

```text
m000_coremark_1core_20000_40000_1ns_20000_22000.final.tog.csr.npz
```

## 10. 推荐运行顺序

建议按如下顺序执行：

```bash
# 1. 编译 fsdbapi
cd /home/yu.zhang/workspace/fsdbapi
make

# 2. 转换两个香山 FSDB，生成 20 个 2000ns 窗口 cache 并 merge，按自己设计的fsdb文件和数量修改shell脚本
./bin/run_xiangshan_2fsdbapi.csh

# 3. 使用 merged manifest 做降维
cd /home/yu.zhang/workspace/powermodeling-demo/step4-dataset-organization/data_prep_new
csh run_fromcache_reduction.csh
