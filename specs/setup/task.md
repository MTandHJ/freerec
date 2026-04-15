# Task List: FreeRec 安装体系重构

- [x] 1. 新建 `pyproject.toml`
- [x] 2. 修改 `freerec/__init__.py`（版本号改为 importlib.metadata + try/except 处理无 torch 场景）
- [x] 3. 修改 `freerec/__main__.py`（新增 setup 子命令，去除 top-level torch 依赖）
- [x] 4. 更新 `README.md` 安装说明
- [x] 5. 删除 `setup.py`
- [x] 6. 验证
