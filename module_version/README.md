# module_version
为项目添加版本查询工具，使用步骤如下
## 1 将module_version做为子项目加入到目标项目中

## 2 在项目根CMmakeLists.txt中使用
```bash
# 推荐用法，project()函数中指定项目名称和项目版本号
# add_version_query(<status>)函数中会自动获取项目名称和版本号
# status: 版本状态，必填参数。例如"develop", "release", "rtm"
set(version_string  1.0.1)
set(version_status develop)
project(test_project_name VERSION ${version_string})
include(./module_version/generate_version_query.cmake)
add_version_query(${version_status})

# 可选用法，若想要另外自行指定项目版本号和项目名称
# add_version_query(<status> [<version_string>] [<module_name>] )
# version_string: 指定版本号
# module_name: 指定模块名
project(test_project_name)
...
set(version_status develop)
include(./module_version/generate_version_query.cmake)
add_version_query(${version_status}  1.0.11 another_name)
```

## 3 控制版本信息是否带git信息
可以通过设置cmake变量VERSION_WHITOUT_GIT来控制工具是否打印项目的git信息
```bash
set(version_string  1.0.1)
set(version_status develop)
project(test_project_name VERSION ${version_string})
include(./module_version/generate_version_query.cmake)
set(VERSION_WHITOUT_GIT 1) # 将VERSION_WHITOUT_GIT变量设置为1， 关闭git信息
add_version_query(${version_status})
```
```bash
#或者在命令行执行cmake的时候传入参数
cmake -DVERSION_WHITOUT_GIT=1 ...
```

## 4 使用方法
在项目安装目录/bin下会出现test_project_name_version可执行文件
```bash
#直接运行会输出当前项目版本信息 
./bin/test_project_name_version

#后面接文件名参数，可以从该文件读取内容，并将本项目的信息新增或覆盖至原信息中
#可用于收集大项目中各子项目的版本信息
./bin/test_project_name_version version.json
```