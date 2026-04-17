# muAlg简介
muAlg 是摩尔线程面向 MUSA 生态提供的并行基础算法头文件库，围绕 GPU 端数据处理场景提供排序、规约、扫描、分区、选择、直方图、块级协作、Warp 级协作等模板化算法组件，可作为上层 AI 计算、图计算、数据分析及科学计算程序的通用算法基础设施。muAlg 基于开源 CUB 1.17 版本进行 MUSA 适配与工程化维护，在保持原有目录结构、命名空间与主要编程接口组织方式的基础上，为 MUSA 应用提供可直接集成的高性能并行算法实现。

# 项目依赖
使用 muAlg 需要安装摩尔线程 GPU 驱动及 MUSA SDK。部分上层接口通常会与 [muThrust](https://github.com/MooreThreads/muThrust) 配套使用，建议同时安装。

# 安装方式
为了方便集成和部署，仓库提供安装脚本。典型用法如下：
```bash
# 默认安装到 /usr/local/musa
./mt_build.sh -i

# 安装到指定目录，例如 /tmp
./mt_build.sh -i -d /tmp

# 从安装目录卸载，不指定 -d 时默认从 /usr/local/musa 卸载
./mt_build.sh -u
```

# 开发者指南
muAlg 基于上游 CUB 1.17 版本维护。关于构建方法、测试流程、平台约束和已知限制，请参考仓库内开发文档与测试说明。
