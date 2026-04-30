"""エントリポイント。

CLIとして直接実行する場合と、
OpenClawのSKILL.mdから呼び出される場合の両方に対応する。

OpenClawからの呼び出し例（SKILL.md内）:
    python3 /path/to/agent/main.py --prompt "{{user_message}}"

環境変数:
    OLLAMA_MODEL: 使用するOllamaモデル（デフォルト: qwen2.5:7b）
    OLLAMA_BASE_URL: OllamaサーバーURL（デフォルト: http://localhost:11434）
    USE_MOCK: "1" を設定するとモッククライアントを使用（テスト・CI用）
"""
import argparse
import os
import sys

from agent.core import Agent, AgentConfig
from agent.llm.base import BaseLLMClient
from agent.llm.mock import MockLLMClient


def build_llm_client() -> BaseLLMClient:
    """環境変数に基づいてLLMクライアントを選択して返す。

    USE_MOCK=1 の場合はモックを返す（CI・テスト用）。
    それ以外はOllamaクライアントを返す。

    Returns:
        設定済みのLLMクライアント。
    """
    if os.getenv("USE_MOCK") == "1":
        return MockLLMClient(response="[mock] LLM is not connected.")

    # Ollamaはimport時にhttpxが必要なため遅延インポート
    from agent.llm.ollama import OllamaClient

    return OllamaClient(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def run_single(prompt: str) -> str:
    """1回の推論を実行して結果を返す。

    OpenClawのSKILL.mdからのコマンドディスパッチ用。

    Args:
        prompt: ユーザーからの入力テキスト。

    Returns:
        エージェントの返答テキスト。
    """
    llm = build_llm_client()
    agent = Agent(llm=llm, config=AgentConfig())
    return agent.run(prompt)


def run_interactive() -> None:
    """インタラクティブなCLIループを実行する。

    ローカルでの動作確認用。
    """
    llm = build_llm_client()
    agent = Agent(llm=llm, config=AgentConfig())
    print("Agent started. Type 'quit' to exit.", file=sys.stderr)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue
        response = agent.run(user_input)
        print(f"Agent: {response}")


def main() -> None:
    """CLIエントリポイント。"""
    parser = argparse.ArgumentParser(description="Local AI Agent")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="単発プロンプト（省略時はインタラクティブモード）",
    )
    args = parser.parse_args()

    if args.prompt:
        # OpenClawから呼ばれるケース: 結果をstdoutに出力
        result = run_single(args.prompt)
        print(result)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
