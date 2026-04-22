# Thrust `async_reduce` known failure 记录（基于 NVIDIA `thrust` tag `1.17.2`）

## 背景

在恢复 `thrust` 默认测试集时，本地 `mp_31` 和远端 `mp_22` 的全量回归都显示：

- `177/177` 个测试程序通过
- `1611/1611` 个测试用例通过
- 另有 `1` 个 `known failure`，不计入失败

该 `known failure` 不是本次 MUSA 适配新引入的问题，而是上游 NVIDIA `thrust` `1.17.2` 就已经在测试源码中显式标注的已知限制。

本记录中的源码摘录均来自：

- 仓库：`NVIDIA/thrust`
- tag：`1.17.2`
- tag commit：`1ac51f2b6219ff17d15d93f2e0be85038556f346`

提取方式示例：

```bash
git -C /data/mingxu/src/cub_1.17/thrust show 1.17.2:testing/async_reduce.cu
```

## 已知失败项

- 测试程序：`thrust.test.async_reduce`
- 具体 case：`test_async_reduce_allocator_on_then_after`
- 现象：测试框架打印 `KNOWN FAILURE`
- 影响：记入 `known failures` 统计，但不会让测试程序或整轮 `ctest` 失败

## 上游源码证据

### 1. 上游 TODO 已把该路径列为 known failure

文件：`testing/async/test_policy_overloads.h`  
tag：`1.17.2`  
行号：`14-19`

```cpp
// TODO Cover these cases from testing/async_reduce.cu:
//   - [x] test_async_reduce_after ("after_future" in test_policy_overloads)
//   - [ ] test_async_reduce_on_then_after (KNOWN_FAILURE, see #1195)
//     - [ ] all the child variants (e.g. with allocator) too
//   - [ ] test_async_copy_then_reduce (Need to figure out how to fit this in)
//   - [ ] test_async_reduce_caching (only useful when returning future)
```

这里已经明确写出：

- `test_async_reduce_on_then_after` 本身就是 `KNOWN_FAILURE`
- 它的 child variants 也包含 allocator 版本

### 2. 具体 allocator 变体在测试体内直接调用 `KNOWN_FAILURE`

文件：`testing/async_reduce.cu`  
tag：`1.17.2`  
行号：`936-997` 中的关键片段

```cpp
auto f2 = thrust::async::reduce(
  thrust::device(thrust::device_allocator<void>{}).on(stream1).after(f1)
, d0.begin(), d0.end()
);

KNOWN_FAILURE;
// FIXME: The below fails because you can't combine allocator attachment,
// `.on`, and `.after`.
// The `#if 0` can be removed once the KNOWN_FAILURE is resolved.
#if 0
  ASSERT_EQUAL_QUIET(stream1, f2.stream().native_handle());
  ...
#endif
```

这段代码给出的结论很直接：

- 失败点正是 `allocator attachment + .on(stream) + .after(future)` 组合
- 该问题在上游 `1.17.2` 就被显式标为 `KNOWN_FAILURE`
- 真正的结果断言被 `#if 0` 屏蔽，等待后续问题解决

对应测试实体定义起点在同文件：

```cpp
struct test_async_reduce_allocator_on_then_after
```

### 3. `KNOWN_FAILURE` 不是普通失败，而是专门的测试状态

文件：`testing/unittest/assertions.h`  
tag：`1.17.2`  
行号：`81-89`

```cpp
#define ASSERT_THROWS(EXPR, EXCEPTION_TYPE)                                   \
  ASSERT_THROWS_WITH_FILE_AND_LINE(EXPR, EXCEPTION_TYPE, __FILE__, __LINE__)  \
  /**/

#define ASSERT_THROWS_EQUAL(EXPR, EXCEPTION_TYPE, VALUE)                                  \
  ASSERT_THROWS_EQUAL_WITH_FILE_AND_LINE(EXPR, EXCEPTION_TYPE, VALUE, __FILE__, __LINE__) \
  /**/

#define KNOWN_FAILURE KNOWN_FAILURE_WITH_FILE_AND_LINE(__FILE__, __LINE__)
```

也就是说，这不是运行时偶发崩溃后被日志“解释”为 known failure，而是测试作者主动在源码里声明的状态。

### 4. 测试框架把 `KnownFailure` 当作允许状态

文件：`testing/unittest/testframework.cu`  
tag：`1.17.2`  
关键行号：`228-246`, `392-393`

```cpp
case KnownFailure:
  std::cout << "KNOWN FAILURE"; num_known_failures++; break;
...
std::cout << num_known_failures << " known failures, ";
```

以及：

```cpp
// all tests pass or are known failures
return true;
```

因此只要某个测试程序中的异常项全部属于 `KnownFailure`，该测试程序整体仍会返回成功。

## 本次 MUSA 回归中的对应现象

### `mp_31`

本地回归日志中，`thrust.test.async_reduce` 输出：

```text
KNOWN FAILURE: test_async_reduce_allocator_on_then_after
Totals: 0 failures, 1 known failures, 0 errors, and 33 passes.
```

同时整轮回归结果为：

- `177/177` 测试程序通过
- `1611/1611` 测试用例通过
- `1` 个 known failure，不计入失败

报告文件：

- `/data/mingxu/src/cub_1.17/thrust/test_report_mp_31.md`

### `mp_22`

远端 `mp_22` 的全量回归结果同样为：

- `177/177` 测试程序通过
- `1611/1611` 测试用例通过
- `1` 个 known failure，不计入失败

报告文件：

- `/data/mingxu/src/cub_1.17/muThrust/test_report_mp_22.md`

## 对 QA 的建议表述

可以直接按下面的口径说明：

> `thrust` 全量回归中存在 1 个 `known failure`，它来自上游 NVIDIA `thrust` tag `1.17.2` 的测试源码预置标记，不是本次 MUSA 变更引入的新回归。  
> 对应 case 为 `thrust.test.async_reduce` 内部的 `test_async_reduce_allocator_on_then_after`。  
> 上游源码已明确注明：当前不支持 `allocator attachment + .on(stream) + .after(future)` 这一组合路径，因此该 case 被框架按 `KnownFailure` 单独统计，但不计入失败。

## 结论

这 1 个 `known failure` 的性质是：

1. 上游 NVIDIA `thrust` `1.17.2` 已知问题
2. 测试源码显式声明，不是运行时新增异常
3. 不影响本轮 `thrust` 全量回归通过结论
4. 若后续要继续调查，应单独针对 `async_reduce` 的 policy 组合限制做专题分析，而不是把它归类为本次测试恢复带来的新回归
