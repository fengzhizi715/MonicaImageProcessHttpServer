# MonicaImageProcessHttpServer


# 一. 说明
该项目是为 https://github.com/fengzhizi715/Monica 项目服务的， 用于部署深度学习模型。
Monica 客户端可以通过调用 http 服务来调用模型，完成推理。


项目目录结构

```
MonicaImageProcess/
├── README.md                        # 项目简介
├── LICENSE                          # 项目许可证文件
├── CMakeLists.txt                   # CMake构建脚本（主）
├── include/                         # 项目公共头文件目录
│   └── cartoon/                     # 图像转换成卡通漫画模块的头文件目录
│   └── common/                      # 封装基类模块的头文件目录
│   └── faceDetect/                  # 人脸识别模块的头文件目录
│   └── faceSwap/                    # 人脸替换模块的头文件目录
│   └── server/                      # http 服务模块的头文件目录
│   └── sketchDrawing/               # 生成素描画模块的头文件目录
│   └── utils/                       # 工具类模块的头文件目录
├── src/                             # 项目主源代码目录
│   └── cartoon/                     # 图像转换成卡通漫画模块
│       └── AnimeGAN.cpp             # 使用 OnnxRuntime 加载 AnimeGANV3 模型的源文件
│   └── common/                      # 封装基类模块
│       └── OnnxRuntimeBase.cpp      # 封装 OnnxRuntime 的基类
│   └── faceDetect/                  # 人脸识别模块
│       └── FaceDetect.cpp           # 使用 OpenCV 的 dnn 模块加载模型实现的人脸识别检测的源文件
│   └── faceSwap/                    # 人脸替换模块
│       └── Face68Landmarks.cpp      # 使用 OnnxRuntime 加载模型实现查找人脸的关键点的源文件
│       └── FaceEmbedding.cpp        # 使用 OnnxRuntime 加载模型实现人脸图像映射的源文件
│       └── FaceEnhance.cpp          # 使用 OnnxRuntime 加载模型实现人脸增强的源文件
│       └── FaceSwap.cpp             # 使用 OnnxRuntime 加载模型实现人脸替换的源文件
│       └── Yolov8Face.cpp           # 使用 OnnxRuntime 加载模型实现人脸检测的源文件
│   └── server/                      # http 服务的模块
│       ├── GlobalResource.cpp       # http 服务加载模型、调用模型的源文件
│       ├── HttpUtils.cpp            # http 服务模块工具类的源文件
│       └── main.cpp                 # http 服务模块主程序入口文件
│   └── sketchDrawing/               # 生成素描画模块
│       └── InformativeDrawings.cpp  # 使用 OnnxRuntime 加载模型实现生成素描画的源文件
│   └── utils/                       # 工具类模块
│       └── Timer.cpp                # 统计某段程序花费时间的工具类
│       └── Utils.cpp                # 常用的工具类
│   └── CMakeLists.txt               # 本地算法的构建脚本
├── models/                          # 存放模型文件的目录
└── .gitignore                       # git 忽略和不追踪的文件
```


# 二. 编译

1. 进入 src 目录：
```
cd src
```

2. 创建一个单独的构建目录
```
mkdir build
cd build
```

3. 使用 CMake 配置项目：
```
cmake ..
```

4. 编译项目
```
cmake --build .
```

# 三. http server

http server 部署了几个模型，方便 Monica 通过调用 http 暴露的服务完成调用这些模型。

## 3.1 http server 的运行
运行编译后的程序

```
Tony-MacBook-Pro:build tony$ ./MonicaImageProcessHttpServer --help
Allowed options:
  -h [ --help ]                         Display help message
  -p [ --http-port ] arg (=8080)        HTTP server port
  -t [ --num-threads ] arg (=16)        Number of worker threads
  -m [ --model-dir ] arg (=/Users/Tony/CLionProjects/MonicaImageProcessHttpServer/models)
                                        Path to the model directory
  -b [ --max-body-size ] arg (=10485760)
                                        Maximum HTTP body size in bytes
```

### Options

| Option                  | Description                                                                                                                                                             |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--num-threads`         | Worker thread pool size.                                                                                                                                                |
| `--http-port`           | HTTP server port.<br/>Default: `8080`                                                                                                                                   |
| `--max-body-size`       | HTTP/HTTPS request payload size limit.<br />Default: 1024 * 1024 * 10(10MB)`                                                                                            |
| `--model-dir`           | Model directory path<br/> |


服务器启动:
```
./MonicaImageProcessHttpServer --http-port 8080 --num-threads 4 --model-dir /Users/Tony/CLionProjects/MonicaImageProcessHttpServer/models
```

## 3.2 接口

### 3.2.0 /health
服务器状态检测

### 3.2.1 /api/sketchDrawing
提供生成素描画的服务

curl 调用的示例:
```
curl -X POST http://localhost:8080/api/sketchDrawing -H "Content-Type: image/jpeg" --data-binary "@/Users/Tony/xxx.png" --output output.jpg
```

## 3.2.2 /api/faceDetect
提供人脸识别的服务

curl 调用的示例:
```
curl -X POST http://localhost:8080/api/faceDetect -H "Content-Type: image/jpeg" --data-binary "@/Users/Tony/xxx.png" --output output.jpg
```

## 3.2.3 /api/faceLandMark
提供人脸检测的服务

curl 调用的示例:
```
curl -X POST http://localhost:8080/api/faceLandMark -H "Content-Type: image/jpeg" --data-binary "@/Users/Tony/xxx.png" --output output.jpg
```

## 3.2.4 /api/faceSwap
提供人脸替换的服务

curl 调用的示例:
```
curl -X POST "http://localhost:8080/api/faceSwap" -H "Content-Type: multipart/form-data" -F "src=@/Users/Tony/src.jpg" -F "target=@/Users/Tony/target.jpg" --output output.jpg
```

## 3.2.5 /api/cartoon
将图像转换成不同的漫画卡通风格

curl 调用的示例:
```
curl -X POST "http://localhost:8080/api/cartoon?type=1" -H "Content-Type: image/jpeg" --data-binary "@/Users/Tony/src.jpg" --output output.jpg
```

# 四. 深度学习的模型
存放在当前项目的 /models 文件夹下

> 目前，有三个模型没有提交到 github，主要是太大了。每个都超过了 100 M 我把他们放到百度网盘
>
> 链接: https://pan.baidu.com/s/15XhVHKi-vPGjB2hYa33v_A?pwd=d9mm
>
> 链接: https://pan.baidu.com/s/1cZvMSuOGxl8CuyHdJM27kw?pwd=9mun
>
> 链接: https://pan.baidu.com/s/10bIlCT08XPtUd_GBOGH6cw?pwd=xu4f
>
> 模型下载下来，存放在工程的 /models/ 目录下


# 五. TODO List:
* 增加日志模块
* 增加 json 模块
