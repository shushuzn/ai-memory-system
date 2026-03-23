# AI Memory System

Local-first memory system for AI agents. 轻量、本地优先的 AI 记忆系统。

## Features

- **短期记忆** - 内存缓存，支持 LRU 淘汰和 TTL 过期
- **长期记忆** - 持久化存储，支持关键词检索
- **记忆检索** - 统一接口，支持跨短/长期记忆搜索
- **记忆蒸馏** - 使用本地 LLM 压缩和提炼记忆
- **RAG 支持** - 生成上下文字符串，方便注入 LLM prompt

## Installation

```bash
pip install ai-memory-system
```

Or install from source:

```bash
git clone https://github.com/your-username/ai-memory-system.git
cd ai-memory-system
pip install -e .
```

## Quick Start

```python
from ai_memory_system import MemorySystem

# Initialize
ms = MemorySystem()

# Add memories
ms.add("user_name", "Alice", memory_type="short")
ms.add("last_project", "AI Memory System", memory_type="long")

# Search
results = ms.search("alice")
for r in results:
    print(f"[{r['source']}] {r['value']}")

# Get context for LLM
context = ms.get_context("user preferences")

# Distill memories
distilled = ms.distill()

# Persist
ms.save()
```

## Architecture

```
┌─────────────────────────────────────────┐
│           MemorySystem                   │
│  (Unified interface)                    │
└───────────┬──────────────┬──────────────┘
            │              │
    ┌───────▼──────┐  ┌───▼───────────┐
    │ ShortTerm    │  │ LongTerm      │
    │ (Memory)     │  │ (JSON file)   │
    └──────────────┘  └───────────────┘
            │              │
    ┌───────▼──────────────▼──────┐
    │      MemoryRetriever         │
    │  (Unified search)            │
    └──────────────────────────────┘
                    │
           ┌────────▼────────┐
           │   MemoryDistiller │
           │ (LLM compression) │
           └──────────────────┘
```

## Configuration

环境变量:

- `LOCAL_LLM_MODEL` - LLM 模型 (default: qwen2.5:1.5b)
- `LOCAL_LLM_BASE_URL` - LLM API 地址 (default: http://localhost:11434)

## Project Structure

```
ai_memory_system/
├── ai_memory_system/   # Main package
│   ├── __init__.py
│   ├── short_term.py        # Short-term memory (in-memory LRU cache)
│   ├── long_term.py         # Long-term memory (JSON file storage)
│   ├── retrieval.py         # Unified retrieval interface
│   ├── distiller.py         # LLM-based memory distillation
│   ├── memory_system.py     # Core MemorySystem class
│   ├── agent_tool.py        # Agent tool wrapper
│   └── openclaw_integration.py  # OpenClaw workspace integration
├── tests/
│   ├── __init__.py
│   ├── test_short_term.py
│   ├── test_long_term.py
│   ├── test_memory_system.py
│   └── test_agent_tool.py
├── README.md
├── setup.py
├── LICENSE
└── .gitignore
```

## Requirements

- Python 3.10+
- requests (for LLM API calls)
- Local LLM server (Ollama recommended)

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.
