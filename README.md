# 面部风格分析系统

这是一个服务器本地推理的实验性视觉分析工具。照片会上传到本站服务器，由本地 MediaPipe 模型处理；程序不调用第三方大模型 API，也不持久化保存原图。

报告把面部关键点比例与照片条件分开呈现，并给出“主风格 × 次风格”、照片参考度和个性化妆发穿搭建议。综合分由规则指标映射到 5.0–9.5 的实验性产品量表，不是客观颜值、人群排名或医学结论。

报告页可生成 5 张 1080×1440 的 3:4 分享图，并可在导出前隐藏本人照片。

## 本地运行

1. 使用 Python 3.12 创建虚拟环境并安装 `server/requirements.txt`。
2. 按 `server/models/README.md` 下载离线模型。
3. 在项目根目录设置环境变量并启动：

```powershell
$env:MODEL_PATH = "$PWD\server\models\face_landmarker.task"
$env:INDEX_PATH = "$PWD\index.html"
& .\.venv\Scripts\uvicorn.exe server.app:app --host 127.0.0.1 --port 8812
```

打开 `http://127.0.0.1:8812/`。

## 验证

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖正常 JPG/PNG/WebP、伪装格式、动画 WebP、像素上限、多脸拒绝、低质量与大角度照片拦截、姿态评分、风格双倾向与报告接口契约。

真实浏览器冒烟测试需要设置 `TEST_IMAGE`、`CHROME_PATH`，并让 Node 能找到 Playwright，然后运行 `node tests/browser-smoke.cjs`。可选设置 `SCREENSHOT_DIR`，保存上传、预览、手机报告和桌面报告截图。
