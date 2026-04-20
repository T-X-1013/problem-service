# Problem Service

面向问题处理场景的 Python 服务集合，当前仓库已经落地两个可运行的 FastAPI 子服务：

- `extract_service`：从客服对话中抽取客户问题
- `classify_service`：对已抽取的问题做分类，并补齐对应客服回答

另外保留了一个待实现目录：

- `answer_validity_service`：回答有效性判断，当前仅占位

## 当前状态

| 服务 | 状态 | 默认端口 | 说明 |
| --- | --- | --- | --- |
| `services/extract_service` | 已实现 | `9001` | 支持 `mock` / `hf` 两种模式 |
| `services/classify_service` | 已实现 | `9002` | 支持 RAG 检索、`mock` / `hf` 两种模式、CPU fallback |
| `services/answer_validity_service` | 预留 | - | 当前只有说明文档，没有服务入口 |

说明：

- 根目录 `main.py` 目前仍是兼容入口，启动的是问题提取服务。
- 分类服务需要单独从 `services/classify_service/app/main.py` 启动。
- 根目录 `requirements.txt` 目前只转发到 `services/extract_service/requirements.txt`，如果要运行分类服务，请额外安装 `services/classify_service/requirements.txt`。

## 目录结构

```text
problem-service/
  main.py
  requirements.txt
  services/
    extract_service/
      app/
        __init__.py
        main.py
      __init__.py
      README.md
      requirements.txt
    classify_service/
      app/
        __init__.py
        main.py
      __init__.py
      README.md
      requirements.txt
    answer_validity_service/
      README.md
```

## 环境准备

推荐使用 Python 3.10 及以上，并单独准备虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果你只需要运行提取服务：

```powershell
pip install -r requirements.txt
```

如果你还需要运行分类服务：

```powershell
pip install -r services\classify_service\requirements.txt
```

当前两个已实现服务的依赖基本一致，核心依赖包括：

- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `transformers`
- `peft`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `torch`

## 启动方式

### 1. 启动提取服务

推荐直接启动子服务入口：

```powershell
.\.venv\Scripts\Activate.ps1
$env:EXTRACTOR_MODE = "mock"
uvicorn services.extract_service.app.main:app --host 127.0.0.1 --port 9001
```

也可以走根目录兼容入口：

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

这两种方式最终启动的都是提取服务。

### 2. 启动分类服务

```powershell
.\.venv\Scripts\Activate.ps1
$env:CLASSIFIER_MODE = "mock"
uvicorn services.classify_service.app.main:app --host 127.0.0.1 --port 9002
```

## 运行模式

### 提取服务 `extract_service`

- `EXTRACTOR_MODE=mock`
  不加载真实模型，使用规则化 mock 逻辑返回 1 到 3 个问题，适合先联调 HTTP 链路。
- `EXTRACTOR_MODE=hf`
  加载 Hugging Face 模型和 LoRA adapter，走真实抽取链路。

常用环境变量：

- `HOST`
- `PORT`，默认 `9001`
- `BASE_MODEL_DIR`
- `ADAPTER_DIR`，默认 `E:\hangdong_tool\model\extraction`
- `MAX_NEW_TOKENS`，默认 `256`
- `RETURN_RAW`，设为 `true` 时返回模型原始输出
- `MAX_RETRIES`，默认 `1`

`hf` 模式示例：

```powershell
$env:EXTRACTOR_MODE = "hf"
$env:BASE_MODEL_DIR = "E:\models\Qwen3-8B"
$env:ADAPTER_DIR = "E:\hangdong_tool\model\extraction"
$env:RETURN_RAW = "true"
uvicorn services.extract_service.app.main:app --host 127.0.0.1 --port 9001
```

### 分类服务 `classify_service`

- `CLASSIFIER_MODE=mock`
  不调用生成模型，但仍会执行本地 RAG 检索，并使用 top-1 检索结果生成分类结果。
- `CLASSIFIER_MODE=hf`
  对每个问题执行完整分类链路：RAG 检索、拼接 prompt、调用 Hugging Face 模型、清洗 JSON、必要时回退。

常用环境变量：

- `HOST`
- `PORT`，默认 `9002`
- `BASE_MODEL_DIR`
- `ADAPTER_DIR`
- `KNOWLEDGE_DIR`
- `RAG_TOP_K`，默认 `3`
- `RAG_MIN_SCORE`，默认 `1.0`
- `RAG_CONTEXT_LIMIT`，默认 `3000`
- `MAX_NEW_TOKENS`，默认 `384`
- `RETURN_RAW`，设为 `true` 时返回检索轨迹或模型原始输出
- `HF_ALLOW_CPU`，默认 `false`

说明：

- `ADAPTER_DIR` 默认会优先探测 `E:\hangdong_tool\model\classification`，找不到时再尝试 `E:\hangdong_tool\model\classifacation`。
- 当 `CLASSIFIER_MODE=hf` 且运行环境没有可用 CUDA 时：
  - 如果 `HF_ALLOW_CPU=false`，服务会自动退回到基于 RAG 的分类结果；
  - 如果 `HF_ALLOW_CPU=true`，服务会尝试在 CPU 上加载模型，但速度可能非常慢。

`hf` 模式示例：

```powershell
$env:CLASSIFIER_MODE = "hf"
$env:BASE_MODEL_DIR = "E:\models\Qwen3-8B"
$env:ADAPTER_DIR = "E:\hangdong_tool\model\classification"
$env:RETURN_RAW = "true"
uvicorn services.classify_service.app.main:app --host 127.0.0.1 --port 9002
```

## 接口概览

### 提取服务

- `GET /health`
- `POST /extract`

请求体：

```json
{
  "info": "客服：您好。\n客户：为什么后续还要收费？"
}
```

返回示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": [
    {
      "问题": "是否会额外收费",
      "原文摘要": "为什么后续还要收费？",
      "解释": "客户确认后续是否存在额外资费"
    }
  ],
  "model": "mock-extractor",
  "latencyMs": 12
}
```

### 分类服务

- `GET /health`
- `POST /classify`

请求体支持两种问题输入方式，优先使用 `problems`，为空时再解析 `problem`：

```json
{
  "info": "客服：您好。\n客户：为什么后续还要收费？\n客服：不会额外收费，只是套餐说明。",
  "problem": "[{\"问题\":\"是否会额外收费\",\"原文摘要\":\"为什么后续还要收费？\",\"解释\":\"客户担心存在额外收费\"}]",
  "problems": [
    {
      "问题": "是否会额外收费",
      "原文摘要": "为什么后续还要收费？",
      "解释": "客户担心存在额外收费"
    }
  ]
}
```

返回示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": [
    {
      "针对的问题": "是否会额外收费",
      "问题大类编号": "05",
      "问题大类名称": "询问是否需要额外收费",
      "问题小类编号": "003",
      "问题小类名称": "长期与期限费用类",
      "客服回答": "不会额外收费，只是套餐说明。",
      "原文摘要": "为什么后续还要收费？",
      "解释": "客户担心存在额外收费"
    }
  ],
  "model": "local-rag-mock-classifier",
  "effectiveMode": "mock",
  "latencyMs": 18
}
```

## 健康检查

提取服务：

```powershell
Invoke-RestMethod http://127.0.0.1:9001/health
```

分类服务：

```powershell
Invoke-RestMethod http://127.0.0.1:9002/health
```

分类服务的 `/health` 返回信息会更完整，除了基础状态外，还会包含：

- `effectiveMode`
- `warning`
- `baseModelExists`
- `adapterExists`
- `torchAvailable`
- `cudaAvailable`
- `deviceName`
- `knowledgeDir`
- `knowledgeLoaded`
- `knowledgeCount`

## RAG 知识目录

分类服务启动后会优先从以下位置寻找知识库目录：

1. `KNOWLEDGE_DIR`
2. `services/classify_service/knowledge/CustomerObjectionClassification-md`
3. `document/CustomerObjectionClassification-md`
4. `src/main/resources/document/CustomerObjectionClassification-md`
5. `D:\project\mutliAgent\multi-agent-service\src\main\resources\document\CustomerObjectionClassification-md`

知识文件格式来自 Java 项目使用的分类 Markdown 文档。

## 与 Java 联调

提取服务配置示例：

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

分类服务配置示例：

```yaml
problem:
  classify:
    mode: custom
    primary: custom
    fallback-to-llm: true
    custom:
      url: http://127.0.0.1:9002/classify
      connect-timeout-ms: 2000
      read-timeout-ms: 120000
```

## 建议的调试顺序

1. 先用两个服务的 `mock` 模式打通 Python HTTP 链路。
2. 手工验证 `/health`、`/extract`、`/classify`。
3. 再切到 `hf` 模式验证模型路径、LoRA adapter 和 GPU 环境。
4. 最后再联调 Java 侧调用链路。

## 子服务文档

如果需要更细的运行说明，可以分别查看：

- `services/extract_service/README.md`
- `services/classify_service/README.md`
- `services/answer_validity_service/README.md`
