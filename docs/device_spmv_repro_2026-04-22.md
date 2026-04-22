# device_spmv 定向复现记录（2026-04-22）

## 目的

在不重新放开完整 `device_spmv` 测试矩阵的前提下，先用一个独立的 standalone repro 稳定复现当前 `mp_31` / MUSA 上的两个问题面：

1. 单个 merge tile、仅命中 `spmv_kernel` 的错误路径。
2. 多个 merge tile、会继续命中 `segment_fixup_kernel` 的错误路径。

## 环境

- 仓库：`/data/mingxu/src/cub_1.17/cub`
- 基线 commit：`ddb3e951`
- 日期：`2026-04-22`
- GPU：`GPU 4 = MTT S5000`
- 架构：`mp_31`
- 测试方式：单进程、单 GPU、本地串行执行

## 当前 dirty state

本次实验不是在干净 worktree 上进行，而是基于当前探索性修改后的状态：

- `cub/device/dispatch/dispatch_spmv_orig.cuh:509`
  - 对 `num_rows > 0 && num_cols == 0` 的 empty problem 路径补了 `musaMemset(d_vector_y, 0, ...)`。
- `cub/agent/agent_spmv_orig.cuh:642`
  - 在 `DIRECT_LOAD_NONZEROS && num_merge_tiles == 1` 分支增加了一次把 `tile_carry` 直接刷回 `d_vector_y` 的尝试。
- `test/CMakeLists.txt:83`
  - 暂时只排除了原始 `test_device_spmv.cu`，保留 `test_device_spmv_no_thrust.cu`，便于后续继续调查。

这意味着下面的复现结果可以回答的问题是：

- 上述探索性 patch 之后，`device_spmv` 的核心错误是否仍然存在。

不能直接回答的问题是：

- 干净基线下的完整失败矩阵是否完全一致。

## 定向 repro 源

- 源文件：`repros/device_spmv/repro_device_spmv_cases.cu`
- 关键点：
  - 使用固定随机种子 `42`
  - `single_tile_double` 对应 `double / 16x16 / fill=0.5`
  - `segment_fixup_float` 对应 `float / 100x100 / fill=0.1`
  - `DeviceSpmv::CsrMV(..., debug_synchronous=true)`，让 kernel launch 直接打印出来

## 编译命令

```bash
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_31 -std=c++17 \
  -I/data/mingxu/src/cub_1.17/cub \
  -I/data/mingxu/src/cub_1.17/cub/test \
  -I/usr/local/musa/include \
  /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases.cu \
  -lmusart \
  -o /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases
```

编译结果：

- 成功生成可执行文件。
- 有两组来自 `cub/block/block_load.cuh:169` 的 `-Wsizeof-array-div` warning。
- warning 同时出现在 `float` / `double` 实例化与 host / mp_31 编译阶段，但不影响本次 repro 运行。

## 运行命令

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases --case=single_tile_double
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases --case=segment_fixup_float
```

## 结果

### 1. `single_tile_double`

命令输出：

```text
device_spmv targeted repro
device=0

[case] single_tile_double
rows=16 cols=16 nnz=120 fill_ratio=0.5
Invoking spmv_kernel<<<{1,1,1}, 96, 0, 0>>>(), 4 items per thread, 16 SM occupancy
Mismatch at position 15: ref=-7829.67 got=-3614.27
[result] FAIL
```

观察：

- 只打印了 `spmv_kernel`，没有 `segment_fixup_kernel`。
- 这说明失败仍然落在“单 tile / direct-load 主路径”上。
- 与此前 probe 中 `double 16x16` 的判断一致：这是一个不依赖 `segment_fixup` 的独立错误类。

### 2. `segment_fixup_float`

命令输出：

```text
device_spmv targeted repro
device=0

[case] segment_fixup_float
rows=100 cols=100 nnz=1031 fill_ratio=0.1
Invoking spmv_kernel<<<{2,1,1}, 128, 0, 0>>>(), 7 items per thread, 16 SM occupancy
Invoking segment_fixup_kernel<<<{1,1,1}, 128, 0, 0>>>(), 3 items per thread, 16 SM occupancy
Mismatch at position 19: ref=3535.03 got=-7859.44
[result] FAIL
```

观察：

- 该 case 明确触发了 `segment_fixup_kernel`。
- 错误仍然稳定复现，说明多 tile 结果归并/修正链路上的问题依旧存在。
- 与此前 probe 中 `float 100x100, fill=0.1` 的现象一致。

## 结论

本次定向 repro 在当前 dirty state 下确认了两件事：

1. `device_spmv` 至少仍有两个彼此独立的问题面。
2. 先前针对 empty problem 和 single-tile carry 回写的探索性 patch，并没有消除这两个核心失败：
   - `single_tile_double` 仍失败，问题不依赖 `segment_fixup_kernel`。
   - `segment_fixup_float` 仍失败，问题落在多 tile 的 fixup 路径。

因此，后续分析应继续拆成两条线：

1. 单 tile / direct-load 主路径。
2. 多 tile / `segment_fixup_kernel` 路径。

## 后续建议

1. 保留该 standalone repro，作为后续每次修改后的第一优先回归。
2. 先不要直接撤销 `device_spmv` 屏蔽；至少应先让上述两个 case 中的一个稳定转绿。
3. 如果接下来要继续调试，优先在 `single_tile_double` 上做更小步分析，因为它不依赖 fixup 链路，问题面更窄。

## 后续调试结果补充（同日更新）

在上面的初始记录之后，又继续做了几轮更细的定向调试，最终确认根因不在 `DeviceSpmv` 高层算法本身，而在更底层的 warp mask helper。

### 1. 关键观察

- 对 `single_tile_double` 做 host 侧逐线程仿真后，`tid=32/33` 的输入和 `scan_segment[]` 按源码逻辑本应推导出正确结果。
- 设备侧临时打印表明：
  - `tid=32` 的 pre-scan 输入与 host 仿真完全一致。
  - 但 `BlockScan` 之后，`tid=32` 的 `scan_item` 错误地变成了 `{15, 4215.395710}`。
  - 正确值应为 `{15, 0}`。
- 这说明错误不是出在 `scan_segment[]` 生成阶段，而是出在 warp0 生成其 aggregate 的阶段。

### 2. 真正根因

根因位于 `cub/util_ptx.cuh` 中 MUSA 侧的 lane mask 实现：

- `LaneMaskLe()` 原先实现为：

```cpp
return (1u << (__get_laneid() % 32 + 1)) - 1;
```

- 当 `lane == 31` 时，上式会执行 `1u << 32`，这是未定义行为。
- `WarpScanShfl` 在 `ReduceByKeyOp` 路径里用到了：

```cpp
ballot = ballot & LaneMaskLe();
int segment_first_lane = CUB_MAX(0, 31 - __clz(ballot));
```

- 对 `single_tile_double` 而言，warp0 的最后一个 lane（`tid=31`，即 `lane 31`）恰好就是一个新的 segment 起点。
- `LaneMaskLe()` 在这个边界上出错后，segment 起点计算被破坏，进而把 warp0 aggregate 算错，最终传给 warp1 lane0 的 prefix 也错了。

### 3. 修复

修复文件：

- `cub/util_ptx.cuh`

修复方式：

- 不再用 `(1u << (lane + 1)) - 1` 这种在 `lane 31` 上会溢出的写法。
- 改为基于 `LaneMaskLt()` 和当前 lane 位拼装：

```cpp
LaneMaskLe() = LaneMaskLt() | (1u << lane)
LaneMaskGt() = ~LaneMaskLe()
LaneMaskGe() = ~LaneMaskLt()
```

### 4. 修复后验证

#### standalone repro

命令：

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases --case=all
```

结果：

- `single_tile_double`: PASS
- `segment_fixup_float`: PASS
- `warp_scan_kv_double`: PASS

#### no-thrust 真实测试目标

编译：

```bash
cmake --build /data/mingxu/src/cub_1.17/cub/build_spmv_probe_test_mp31 --target cub.cpp17.test.device_spmv -j1
```

运行：

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/build_spmv_probe_test_mp31/bin/cub.cpp17.test.device_spmv
```

结果：

- `float / double / int` 全部通过
- 覆盖了：
  - `1x0`
  - `16x16, fill=0.5`
  - `100x100, fill=0.1`
  - `256x256, fill=0.3`
- 最终输出：`=== All tests PASS ===`

### 5. 关于“撤销屏蔽”的当前状态

- 已经删除 `test/CMakeLists.txt` 里对 `test_device_spmv.cu` 的显式 `REMOVE_ITEM`。
- 默认构建仍会按现有 CMake 逻辑优先选择 `_no_thrust` 版本。
- 到上一轮记录为止，已经可以确认：
  - `device_spmv` 的核心 warp-scan / segment-boundary 错误已经修复。
  - `no_thrust` 的真实测试路径已经验证通过。

## 6. 原始 thrust 入口重新放开后的补充调查

后续重新用干净 build 目录配置：

```bash
cmake -S /data/mingxu/src/cub_1.17/cub \
  -B /data/mingxu/src/cub_1.17/cub/build_spmv_thrust_orig_mp31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUB_ENABLE_TESTING=ON \
  -DMUSA_ARCH_LIST=mp_31 \
  -DCUB_IGNORE_NO_THRUST_TESTS=ON
```

可以确认原始 thrust-backed `test_device_spmv.cu` 是可以正常生成目标的；此前“目标没出现”不是算法问题，而是那次 build 配置本身不对。

### 6.1 新发现的问题面

原始 thrust 入口重新跑通编译后，又暴露出两个之前 no-thrust 路径没有覆盖到的问题：

1. `double + num_nonzeros == 0`
   - 用例形状：`test_random<double>(98, 84, 0)`
   - 现象：`spmv_kernel` 在同步时报 `illegal memory access`
   - 原因：`DispatchSpmv` 只把 `num_rows == 0 || num_cols == 0` 当成 empty problem，没有把 `num_nonzeros == 0` 视为可直接返回的零矩阵路径。

2. `double + exact single-tile full matrix`
   - 用例形状：`test_random<double>(128, 2, 1.0002)`
   - 现象：最后一个输出元素丢失，`Mismatch at position 127: 1807.69 vs 0`
   - 原因：单个 merge tile 且没有 `segment_fixup_kernel` 时，最后一个未闭合 segment 的 `tile_carry` 没有被正确刷回 `d_vector_y`。

### 6.2 对应修复

修复文件：

- `cub/device/dispatch/dispatch_spmv_orig.cuh`
- `cub/agent/agent_spmv_orig.cuh`
- `test/test_device_spmv.cu`

修复内容：

1. `DispatchSpmv`
   - 把 `num_nonzeros == 0` 也纳入 empty/zero-filled matrix 的短路分支。
   - 若 `num_rows > 0`，直接 `musaMemset(d_vector_y, 0, num_rows * sizeof(ValueT))` 后返回。

2. `AgentSpmv`
   - 单 tile 无 fixup 的场景下，若 `tile_carry.key` 仍指向 tile 内有效 row，就直接把 `tile_carry.value` 写回对应输出行。
   - 这里写回索引使用 `tile_carry.key` 本身，不再做错误的 `key - 1` 偏移，也不再只限制在 `DIRECT_LOAD_NONZEROS` 分支。

3. 原始 thrust 测试文件
   - `friend class csr_matrix<...>` 改成 `friend struct csr_matrix<...>`，修正 `-Wmismatched-tags` 在 MUSA/clang + `-Werror` 下的编译失败。

### 6.3 补充定向 repro

在 `repros/device_spmv/repro_device_spmv_cases.cu` 中新增：

- `zero_nnz_double`
- `single_tile_full_double`

现在 `--case=all` 会覆盖：

- `single_tile_double`
- `segment_fixup_float`
- `zero_nnz_double`
- `single_tile_full_double`
- `warp_scan_kv_double`

### 6.4 最终验证

#### 原始 thrust 入口

编译：

```bash
cmake --build /data/mingxu/src/cub_1.17/cub/build_spmv_thrust_orig_mp31 --target cub.cpp17.test.device_spmv -j1
```

运行：

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/build_spmv_thrust_orig_mp31/bin/cub.cpp17.test.device_spmv
```

结果：

- 完整随机矩阵测试矩阵全部跑完，进程退出码为 `0`

#### 更新后的 standalone repro

运行：

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/repros/device_spmv/repro_device_spmv_cases --case=all
```

结果：

- `single_tile_double`: PASS
- `segment_fixup_float`: PASS
- `zero_nnz_double`: PASS
- `single_tile_full_double`: PASS
- `warp_scan_kv_double`: PASS

#### no-thrust 真实测试目标复验

编译：

```bash
cmake -S /data/mingxu/src/cub_1.17/cub \
  -B /data/mingxu/src/cub_1.17/cub/build_spmv_nothrust_mp31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUB_ENABLE_TESTING=ON \
  -DMUSA_ARCH_LIST=mp_31

cmake --build /data/mingxu/src/cub_1.17/cub/build_spmv_nothrust_mp31 --target cub.cpp17.test.device_spmv -j1
```

运行：

```bash
MUSA_VISIBLE_DEVICES=4 /data/mingxu/src/cub_1.17/cub/build_spmv_nothrust_mp31/bin/cub.cpp17.test.device_spmv
```

结果：

- 最终输出：`=== All tests PASS ===`
- 说明本轮补丁没有把已经转绿的 no-thrust 路径带坏
