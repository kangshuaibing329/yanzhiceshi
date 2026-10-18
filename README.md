# 面部风格分析系统

这是一个服务器本地推理的实验性视觉分析工具。照片会上传到本站服务器，由本地 MediaPipe 模型处理；程序不调用第三方大模型 API，也不持久化保存原图。

报告把面部关键点比例与照片条件分开呈现，并给出“主风格 × 次风格”、混合倾向、照片参考度和个性化妆发穿搭建议。照片呈现分由规则指标映射到 5.0–9.5 的实验性产品量表，只描述当前照片，不是客观颜值、人群排名或医学结论。

报告页可生成 5 张 1080×1440 的 3:4 分享图，并可在导出前隐藏本人照片。

当前算法会把面部关系、照片质感、姿态和构图分开计算，并给出照片参考度与可信度提示；风格标签同时返回主次风格及四类候选信号，便于解释“为什么得到这个标签”。规则权重目前仍未经过足量、脱敏、取得授权的人群数据校准，因此不输出人群百分位，也不用于人与人排名。

报告页在移动端采用章节导航与折叠详情，桌面端展开完整依据；分享弹窗默认隐藏本人照片，支持 5 张报告图切换、键盘焦点回收，以及支持系统分享时的分享 / 下载回退。

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
