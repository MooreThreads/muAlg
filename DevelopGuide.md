# 移植步骤
## 1 仓库准备
下载cub 1.12.1 的release版本。本项目初始基于cub 1.12.1版本。
## 2 musify一键迁移
musify迁移工具的使用教程详见[博客](https://blog.mthreads.com/blog/musa/2024-05-28-%E4%BD%BF%E7%94%A8musify%E5%AF%B9%E4%BB%A3%E7%A0%81%E8%BF%9B%E8%A1%8C%E5%B9%B3%E5%8F%B0%E8%BF%81%E7%A7%BB/)。在项目根目录下执行命令, 即可完成迁移。
```
/usr/local/musa/tools/musify-text --inplace -- `find ./ -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h'`
```
## 3 cmake改造
MUSA cmake的使用教程详见[博客](https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/)。
* CMakeLists.txt:83, 替换成cmake/CubMusaConfig.cmake。添加新文件`cmake/CubMusaConfig.cmake`。用于做MUSA camke的配置。
* `cmake/CubHeaderTesting.cmake`, `examples/CMakeLists.txt`, `examples/cmake/add_subdir/CMakeLists.txt`, `test/CMakeLists.txt`, 等文件中将add_executable改成musa_add_executable, 将add_library改成musa_add_library，为了让代码能使用mcc编译器编译。
## 4 其他
这里记录一些非普适也非所有项目必要的改动。
* 添加muAlg库的安装脚本`mt_build.sh`。
* 添加新的`README.md`和`DeveloperGuide.md`指引文件。
* 为修改的代码文件添加版权许可说明。
* 增加module_version目录，目的是给muAlg库增加版本查询功能。使用见`CMakeLists.txt:27`。

## 5 GPU适配改动
* cub/util_ptx.cuh 注释不支持的内嵌汇编，改为相同功能的builtin函数实现。
* cub/util_arch.cuh:53 MUSA总是使用COOPERATIVE_GROUPS。
* cub/util_device.cuh:380 修改成使用musaGetDeviceProperties函数获取架构号。
* cub/util_type.cuh:382,420 将DeviceWord限制到8bytes。
* cub/warp/specializations/warp_scan_shfl.cuh以及cub/warp/specializations/warp_reduce_shfl.cuh 注释不支持的内嵌汇编，改为相同功能的builtin函数实现。
* cub/thread/thread_load.cuh以及cub/thread/thread_store.cuh 注释不支持的内嵌汇编，改为相同功能的builtin函数实现。
* cub/block/radix_rank_sort_operations.cuh:128,148 增加this->以修复不规范写法。
* cub/agent/single_pass_scan_operators.cuh:122,452 缩小SINGLE_WORD的限制条件,限制到8bytes。
* cub/device/dispatch/dispatch_* 所有文件中头文件路径改成 thrust/system/musa/detail/core/triple_chevron_launch.h 这个是配合muThrust项目的修改。
* cub/device/dispatch/dispatch_* 中的文件，可以自定义Policy以调整device算法的流程用的子步骤的算法和配置。这里给出两个例子cub/device/dispatch/dispatch_scan.cuh:197 以及cub/device/dispatch/dispatch_radix_sort.cuh:940。开发者或用户可以自行调整到性能最佳的可用状态。





