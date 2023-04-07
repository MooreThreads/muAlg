# 安装到本地
```bash
  # 默认安装目录为/usr/local/musa/include
  ./mt_build.sh -i
```

# 安装到本地指定目录
```bash
  # 安装到目录/tmp/include, 注意-d指定目录前缀无需加include
  ./mt_build.sh -i -d /tmp
```

# 打包后用于发布到其他环境安装
```bash
  # 打包到{项目根目录}/build/package/
  ./mt_build.sh -p
```