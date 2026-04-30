"""エージェントのコアロジック。

入力 → 処理 → 出力 のパイプラインを定義する。
LLMクライアントはDIで注入するため、モックへの差し替えが可能。
"""
from dataclasses import dataclass, field

from agent.llm.base import BaseLLMClient, Message


@dataclass
class AgentConfig:
    """エージェントの設定。"""

    system_prompt: str = (
        "You are a helpful assistant. "
        "Answer concisely and accurately."
    )
    max_history: int = 10


@dataclass
class AgentState:
    """エージェントの会話状態。"""

    history: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """メッセージを履歴に追加する。"""
        self.history.append(Message(role=role, content=content))

    def clear(self) -> None:
        """履歴をリセットする。"""
        self.history.clear()


class Agent:
    """LLMを利用するエージェント。

    LLMクライアントをコンストラクタで受け取ることで、
    ローカルLLM・クラウドAPI・モックを差し替え可能にしている。

    将来的な拡張ポイント:
    - ツール呼び出し（関数実行）の追加
    - マルチターン会話の管理
    - 外部フレームワーク（LangChain等）との統合
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        config: AgentConfig | None = None,
    ) -> None:
        """初期化。

        Args:
            llm: LLMクライアント（BaseLLMClientのサブクラス）。
            config: エージェント設定。省略時はデフォルト値を使用。
        """
        self._llm = llm
        self._config = config or AgentConfig()
        self._state = AgentState()

    def run(self, user_input: str) -> str:
        """ユーザー入力を処理して返答を返す。

        Args:
            user_input: ユーザーからのテキスト入力。

        Returns:
            エージェントの返答テキスト。

        Raises:
            ValueError: user_inputが空文字の場合。
        """
        if not user_input.strip():
            raise ValueError("user_input must not be empty")

        self._state.add("user", user_input)

        messages = self._build_messages()
        response = self._llm.complete(messages)

        self._state.add("assistant", response.content)
        self._trim_history()

        return response.content

    def reset(self) -> None:
        """会話履歴をリセットする。"""
        self._state.clear()

    @property
    def history(self) -> list[Message]:
        """現在の会話履歴を返す（読み取り専用）。"""
        return list(self._state.history)

    def _build_messages(self) -> list[Message]:
        """LLMに渡すメッセージリストを構築する。"""
        system = [Message(role="system", content=self._config.system_prompt)]
        return system + self._state.history

    def _trim_history(self) -> None:
        """max_historyを超えた古い履歴を削除する。"""
        max_pairs = self._config.max_history
        if len(self._state.history) > max_pairs * 2:
            self._state.history = self._state.history[-(max_pairs * 2):]
