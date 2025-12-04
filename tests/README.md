# 测试套件说明 (Test Suite Instructions)

This directory contains test scripts for the Monica Image Processing HTTP Server.

## 测试脚本 (Test Scripts)

### 1. Python 测试脚本 (Python Test Script)
- **文件**: `test_server.py`
- **依赖**: Python 3 和 requests 库
- **运行方法**:
  ```bash
  pip install requests
  python test_server.py
  ```

### 2. Shell 测试脚本 (Shell Test Script)
- **文件**: `test_server.sh`
- **依赖**: curl
- **运行方法**:
  ```bash
  chmod +x test_server.sh
  ./test_server.sh
  ```

## 运行测试前的准备 (Preparation)

1. 启动服务器 (Start the server):
   ```bash
   cd ../src/build
   ./MonicaImageProcessHttpServer
   ```

2. (可选) 准备测试图片 (Optional) Prepare a test image:
   - 将一张 JPG 图片命名为 `test_image.jpg` 放在 tests 目录中
   - Place a JPG image named `test_image.jpg` in the tests directory

## 测试的端点 (Tested Endpoints)

- `/health` - 健康检查 (Health check)
- `/version` - 版本信息 (Version info)
- `/api/sketchDrawing` - 素描画效果 (Sketch drawing effect)
- `/api/faceDetect` - 人脸检测 (Face detection)
- `/api/faceLandMark` - 面部关键点检测 (Face landmark detection)
- `/api/cartoon` - 卡通化效果 (Cartoon effect)

## 注意事项 (Notes)

- 如果没有提供测试图片，部分测试会被跳过
- If no test image is provided, some tests will be skipped
- 服务器必须在运行状态下才能执行测试
- Server must be running to execute tests