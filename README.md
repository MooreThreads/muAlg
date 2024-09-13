# muAlg简介
muAlg 库是一个头文件库，提供了GPU端的模板函数实现如排序，规约，前缀和等基础常用算法，让这些计算能运行在GPU上。muAlg是开源库cub的MUSA移植项目，这个项目的目的主要有两点。第一，在GPU加速领域的生态中，有大量的项目使用到了cub库，为了让这些项目能运行在摩尔线程的GPU上，需要进行MUSA移植，muAlg项目便提供了移植好的可以直接使用的cub库。另一个目的是，通过这个项目，给广大开发者提供MUSA移植的实践案例，供大家参考。
# 项目依赖
确保环境中使用摩尔线程GPU，并请访问官网下载和安装[MUSA SDK](https://developer.mthreads.com/sdk/download/musa)。另外muAlg的使用还需要一同安装好[muThrust](https://github.com/MooreThreads/muThrust)库。
# 安装方式
为了使用方便，提供了安装脚本，使用方式如下：
```bash
  # 默认安装目录为/usr/local/musa
  ./mt_build.sh -i

  # 安装到指定目录，例如安装到/tmp
  ./mt_build.sh -i -d /tmp

  # 从安装目录中卸载，不加-d则默认从/usr/local/musa卸载
  ./mt_build.sh -u
```

# 开发者指南
由于cub项目较大，muAlg的移植与针对性优化需要一个过程，欢迎广大开发者参与muAlg库的完善。[开发者指南](./DevelopGuide.md)文档中记录了初版提交的移植步骤以及一些说明。