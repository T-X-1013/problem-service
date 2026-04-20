# Extract Service

问题提取子服务，供 Java 主服务通过 HTTP 调用。

当前目录结构下，这个子服务位于：

```text
E:\hangdong_pros\problem-service\services\extract_service
```

## 目录结构

```text
extract_service
├─ app
│  ├─ __init__.py
│  └─ main.py
├─ requirements.txt
├─ README.md
└─ __init__.py
```

说明：

- `app/main.py`：FastAPI 服务入口
- `requirements.txt`：当前子服务依赖
- `README.md`：启动与联调说明

## 运行前准备

### 1. 代码目录

当前子服务目录：

```text
E:\hangdong_pros\problem-service\services\extract_service
```

### 2. 虚拟环境

推荐做法：

- 代码放在 `E:` 盘
- Python 虚拟环境放在 `C:` 盘
- `extract_service` 和 `classify_service` 共用同一个虚拟环境

创建虚拟环境：

```powershell
python -m venv C:\Users\63338\venvs\problem-python-service
```

激活虚拟环境：

```powershell
C:\Users\63338\venvs\problem-python-service\Scripts\Activate.ps1
```

进入子服务目录并安装依赖：

```powershell
cd E:\hangdong_pros\problem-service\services\extract_service
pip install -r requirements.txt
```

如果这个虚拟环境还要同时运行分类服务，再额外执行一次：

```powershell
cd E:\hangdong_pros\problem-service\services\classify_service
pip install -r requirements.txt
```

### 3. GPU 依赖

如果要运行真实模型，必须安装 GPU 版 PyTorch。

当前已验证可用的一组环境：

- Python 3.10
- `torch 2.11.0+cu128`
- CUDA 12.8
- NVIDIA GeForce RTX 5090

验证命令：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期输出类似：

```text
2.11.0+cu128
True
12.8
NVIDIA GeForce RTX 5090
```

## 模型说明

当前抽取服务的真实模型由两部分组成：

1. base model：`Qwen3-8B`
2. LoRA adapter：`E:\hangdong_tool\model\extraction`

注意：

- `E:\hangdong_tool\model\extraction` 只是 adapter，不是完整模型
- 启动 `hf` 模式时，必须同时提供 `BASE_MODEL_DIR` 和 `ADAPTER_DIR`

base model 路径示例：

```text
E:\models\Qwen3-8B
```

## 运行模式

### 1. mock 模式

适合先和 Java 联调，不依赖真实模型。

```powershell
C:\Users\63338\venvs\problem-python-service\Scripts\Activate.ps1
cd E:\hangdong_pros\problem-service\services\extract_service
pip install -r requirements.txt
$env:EXTRACTOR_MODE = "mock"
uvicorn app.main:app --host 127.0.0.1 --port 9001
```

### 2. hf 模式

适合加载 `Qwen3-8B + LoRA adapter` 的真实模型。

```powershell
C:\Users\63338\venvs\problem-python-service\Scripts\Activate.ps1
cd E:\hangdong_pros\problem-service\services\extract_service
pip install -r requirements.txt
$env:EXTRACTOR_MODE = "hf"
$env:BASE_MODEL_DIR = "E:\models\Qwen3-8B"
$env:ADAPTER_DIR = "E:\hangdong_tool\model\extraction"
$env:RETURN_RAW = "true"
uvicorn app.main:app --host 127.0.0.1 --port 9001
```

说明：

- `EXTRACTOR_MODE=mock`：使用 mock 规则返回结果，适合验证 Java 联调链路
- `EXTRACTOR_MODE=hf`：加载真实模型进行抽取
- `RETURN_RAW=true`：返回模型原始输出，便于调试 JSON 格式问题

## 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:9001/health
```

返回字段说明：

- `mode`：当前运行模式
- `loaded`：真实模型是否已加载
- `loadError`：模型加载失败时的错误信息

注意：

- `hf` 模式下，模型默认懒加载
- 第一次调用 `/extract` 后，`loaded` 才会变成 `true`

## 抽取测试

```powershell
$body = @{
  info = "客服：您好。`n客户：为什么后续还要收费？"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:9001/extract `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

## Java 侧配置

Java 项目中的配置应为：

```yaml
problem:
  extract:
    mode: custom
    primary: custom
    fallback-to-llm: true
    custom:
      url: http://127.0.0.1:9001/extract
      connect-timeout-ms: 2000
      read-timeout-ms: 120000
```

如果只是先联调结构，也可以先用：

```yaml
problem:
  extract:
    mode: dual
    primary: llm
```

## 返回格式

服务返回结构固定为：

```json
{
  "code": 0,
  "message": "ok",
  "data": [
    {
      "问题": "是否会额外收费",
      "原文摘要": "客户：为什么后续还要收费？",
      "解释": "客户确认后续是否存在额外资费"
    }
  ],
  "model": "qwen3-8b-lora-extraction",
  "latencyMs": 1200,
  "raw": "可选，仅在 RETURN_RAW=true 时返回"
}
```

说明：

- `code=0` 表示调用成功
- `data` 是标准化后的问题数组
- `model` 用于区分当前是否在使用真实模型
- `raw` 用于调试模型原始输出

## 当前建议的调试顺序

1. 首次开发先用 `mock` 模式跑通 Python -> Java 联调
2. 代码没问题后，后续就不需要执行 `mock` 模式了，直接切 `hf` 模式验证真实模型加载
3. 手工测试 `/extract`
4. 最后再运行 Java 里的：
   - `CustomModelProblemExtractorTest`
   - `MultiAgentServiceTest`
