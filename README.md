# Problem Service

问题处理大服务根目录。

当前已经落地的能力只有问题提取，后续会继续在同一层级扩展：

- 问题提取
- 问题分类
- 回答是否有效

## 当前目录结构

```text
problem-extract-service/
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
      result.json
    classify_service/
      README.md
    answer_validity_service/
      README.md
```

说明：

- 根目录现在表示“大服务”
- `services/extract_service` 是当前已实现的子服务
- `services/classify_service` 和 `services/answer_validity_service` 目前只保留占位
- 根目录 `main.py` 是兼容入口，当前默认仍启动问题提取服务

## 启动方式

推荐直接启动提取子服务：

```powershell
C:\Users\63338\venvs\problem-extract-service\Scripts\Activate.ps1
cd E:\hangdong_pros\problem-extract-service
pip install -r requirements.txt
uvicorn services.extract_service.app.main:app --host 127.0.0.1 --port 9001
```

兼容旧方式：

```powershell
python main.py
```

## 后续扩展建议

后续新增问题分类或回答有效性判断时，直接在 `services/` 下新增对应子服务目录，并保持每个子服务独立维护：

- 自己的 `app/`
- 自己的 `requirements.txt`
- 自己的 README
- 自己的模型或配置说明
