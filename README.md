# local-ai-agent

ローカルLLM（Ollama）とOpenClawを組み合わせたプライベートAIエージェント。
クラウド依存なし。データはデバイス外に出ない。

## 構成

```
local-ai-agent/
├── agent/
│   ├── main.py        # エントリポイント（CLI / OpenClaw Skill）
│   ├── core.py        # エージェントロジック
│   └── llm/
│       ├── base.py    # LLM抽象クラス
│       ├── mock.py    # テスト用モック
│       └── ollama.py  # Ollama接続（ローカルLLM）
├── openclaw/
│   ├── SKILL.md       # OpenClawスキル定義
│   └── SOUL.md        # エージェント人格定義
└── tests/
```

## セットアップ

```bash
# 1. 依存インストール
pip install -e ".[dev]"

# 2. 環境変数設定
cp .env.example .env
# .env を編集

# 3. Ollamaセットアップ（初回のみ）
ollama pull qwen2.5:7b

# 4. 動作確認
python -m agent.main --prompt "hello"
```

## OpenClaw連携

```bash
# OpenClaw workspaceにSkillをコピー
cp openclaw/SKILL.md ~/.openclaw/workspace/skills/local-ai-agent/SKILL.md
cp openclaw/SOUL.md ~/.openclaw/workspace/SOUL.md

# OpenClaw経由で呼び出し（TUI or Telegram等）
# /local-ai-agent hello from OpenClaw
```

## テスト実行

```bash
pytest
```

## CI/CD

GitHub Actionsで以下を自動実行（PR / mainへのpush）:
- flake8（コード品質）
- mypy（型チェック）
- pytest（`USE_MOCK=1` でLLM呼び出しをスキップ）
