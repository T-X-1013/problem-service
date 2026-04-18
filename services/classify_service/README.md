# Classify Service

问题分类子服务，供 Java 主服务通过 HTTP 调用。

这个服务现在实现的是“真正的 Python RAG 分类”链路，而不是简单规则匹配：

- 读取分类知识 Markdown
- 对每个问题单独做检索
- 将检索到的分类知识注入单题分类 prompt
- 逐题输出并合并标准化结果
- 请求/返回结构尽量对齐 Java 的 `LlmProblemClassifier`

## 目录结构

```text
classify_service
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

## 接口

- `GET /health`
- `POST /classify`

请求体字段：

```json
{
  "info": "客服与客户对话全文",
  "problem": "[{\"问题\":\"...\",\"原文摘要\":\"...\",\"解释\":\"...\"}]",
  "problems": [
    {
      "问题": "...",
      "原文摘要": "...",
      "解释": "..."
    }
  ]
}
```

说明：

- `info`：完整通话文本
- `problem`：Java 侧原样透传的问题 JSON 字符串
- `problems`：Java 侧已解析后的问题对象或数组
- 优先使用 `problems`；为空时回退解析 `problem`

返回结构固定为：

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
      "问题小类名称": "长期与期限费用类：询问长期使用、后续及特定期限后的费用",
      "客服回答": "不会额外收费，只是套餐说明。",
      "原文摘要": "客户：为什么后续还要收费？",
      "解释": "客户担心后续出现额外收费"
    }
  ],
  "model": "local-rag-mock-classifier",
  "latencyMs": 12,
  "raw": "可选，仅在 RETURN_RAW=true 时返回"
}
```

## 行为对齐 Java

当前实现尽量按 Java `LlmProblemClassifier` 的方式处理：

1. 先清洗 `problem` JSON
2. 如果是数组，则逐个问题单独分类
3. 每个问题独立检索 RAG 知识
4. 每个问题独立产出 JSON 分类结果
5. 最终把多题结果合并成一个数组返回

和 Java 相比，Python 版额外做了两层稳健性处理：

- 模型输出不是完整 JSON 时，尝试 salvage 局部对象
- 仍然失败时，可回退到 top-1 检索命中的分类结果

## RAG 知识来源

默认会从以下位置中找到第一个存在的分类知识目录：

1. `KNOWLEDGE_DIR`
2. 当前服务目录下的 `knowledge/CustomerObjectionClassification-md`
3. 当前工作目录下的 `document/CustomerObjectionClassification-md`
4. 当前工作目录下的 `src/main/resources/document/CustomerObjectionClassification-md`
5. `D:\project\mutliAgent\multi-agent-service\src\main\resources\document\CustomerObjectionClassification-md`

知识文件格式来自 Java 项目现有的 `Classification-*.md` 文档。

## 运行前准备

### 1. 代码目录

当前子服务目录：

```text
E:\hangdong_pros\problem-service\services\classify_service
```

### 2. 虚拟环境

推荐做法：

- 代码放在 `E:` 盘
- Python 虚拟环境放在 `C:` 盘

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
cd E:\hangdong_pros\problem-service\services\classify_service
pip install -r requirements.txt
```

这个虚拟环境建议与 `extract_service` 共用。首次初始化时，再补一次：

```powershell
cd E:\hangdong_pros\problem-service\services\extract_service
pip install -r requirements.txt
```

### 3. GPU 依赖

如果要运行真实模型，必须安装 GPU 版 PyTorch。

已验证可用的一组环境：

- Python 3.10
- `torch 2.11.0+cu128`
- CUDA 12.8
- NVIDIA GeForce RTX 5090

验证命令：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 模型说明

当前分类服务真实模型由两部分组成：

1. base model：`Qwen3-8B`
2. LoRA adapter：`E:\hangdong_tool\model\classification`

注意：

- `E:\hangdong_tool\model\classification` 只是 adapter，不是完整模型
- 启动 `hf` 模式时，必须同时提供 `BASE_MODEL_DIR` 和 `ADAPTER_DIR`

base model 路径示例：

```text
E:\models\Qwen3-8B
```

## 运行模式

### 1. mock 模式

`mock` 模式仍然会执行真实 RAG 检索，只是不调用生成模型，而是直接使用检索 top-1 作为分类结果。

适合先与 Java 联调，快速验证：

- 请求结构是否对齐
- 知识库能否正常加载
- 检索结果是否合理
- 客服回答抽取是否正常

```powershell
C:\Users\63338\venvs\problem-python-service\Scripts\Activate.ps1
cd E:\hangdong_pros\problem-service\services\classify_service
pip install -r requirements.txt
$env:CLASSIFIER_MODE = "mock"
uvicorn app.main:app --host 127.0.0.1 --port 9002
```

### 2. hf 模式

`hf` 模式会对每个问题执行完整链路：

1. 检索 top-k 分类知识
2. 将知识内容注入 prompt
3. 调用 `Qwen3-8B + LoRA adapter`
4. 清洗并标准化模型输出 JSON
5. 解析失败时尝试 salvage
6. 仍失败时回退到检索结果兜底

```powershell
C:\Users\63338\venvs\problem-python-service\Scripts\Activate.ps1
cd E:\hangdong_pros\problem-service\services\classify_service
pip install -r requirements.txt
$env:CLASSIFIER_MODE = "hf"
$env:BASE_MODEL_DIR = "E:\models\Qwen3-8B"
$env:ADAPTER_DIR = "E:\hangdong_tool\model\classifacation"
$env:RETURN_RAW = "true"
uvicorn app.main:app --host 127.0.0.1 --port 9002
```

## 常用环境变量

- `PORT`
  默认 `9002`
- `CLASSIFIER_MODE`
  `mock` 或 `hf`
- `KNOWLEDGE_DIR`
  指定分类知识目录
- `RAG_TOP_K`
  默认 `3`
- `RAG_MIN_SCORE`
  默认 `1.0`
- `RAG_CONTEXT_LIMIT`
  默认 `3000`
- `RETURN_RAW`
  设为 `true` 时返回检索轨迹或模型原始输出

## 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:9002/health
```

返回字段说明：

- `mode`：当前运行模式
- `loaded`：真实模型是否已加载
- `loadError`：模型加载失败时的错误信息
- `knowledgeDir`：当前使用的知识目录
- `knowledgeLoaded`：知识库是否加载成功
- `knowledgeCount`：已解析的分类知识条数

## 分类测试

```powershell
$body = @{
  info = "客服：您好。`n客户：为什么后续还要收费？`n客服：不会额外收费，只是套餐说明。"
  problems = @(
    @{
      问题 = "是否会额外收费"
      原文摘要 = "客户：为什么后续还要收费？"
      解释 = "客户担心后续存在额外收费"
    }
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri http://127.0.0.1:9002/classify `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

## Java 侧配置

Java 项目中的配置建议为：

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

## 当前建议的调试顺序

1. 先用 `mock` 模式跑通 Python -> Java 联调
2. 手工验证 `/health` 和 `/classify`
3. 再切换到 `hf` 模式验证真实模型
4. 最后运行 Java 侧测试：
   `CustomModelProblemClassifierTest`

## CPU Fallback

从这版开始，`CLASSIFIER_MODE=hf` 会先检查运行环境：

- 如果检测到 CUDA 可用，继续使用 `Qwen3-8B + LoRA adapter`
- 如果检测不到 CUDA，且 `HF_ALLOW_CPU=false`，服务会自动退回到基于 RAG 检索的分类结果，不再硬跑 8B 模型
- 如果明确设置 `HF_ALLOW_CPU=true`，服务仍会在 CPU 上加载模型，但速度可能非常慢
- 如果未显式设置 `ADAPTER_DIR`，服务会自动优先识别 `classification`，找不到时回退识别本机现有的 `classifacation`

新增健康检查字段：

- `effectiveMode`
- `warning`
- `torchAvailable`
- `cudaAvailable`
- `deviceName`
- `hfAllowCpu`
- `baseModelExists`
- `adapterExists`

新增环境变量：

- `HF_ALLOW_CPU`
  默认 `false`
  不建议在 Java 联调时开启；否则很容易因为 CPU 推理过慢导致 `read-timeout-ms` 超时
