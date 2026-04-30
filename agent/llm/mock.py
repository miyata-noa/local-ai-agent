"""テスト・開発用のモックLLMクライアント。

CI環境やLLM未接続時に使用する。
実際のLLMを呼ばず、設定したレスポンスを返す。
"""
from agent.llm.base import BaseLLMClient, LLMResponse, Message


class MockLLMClient(BaseLLMClient):
    """固定レスポンスを返すモッククライアント。

    テストや開発初期段階での使用を想定。
    """

    def __init__(self, response: str = "mock response") -> None:
        """初期化。

        Args:
            response: completeメソッドが返す固定テキスト。
        """
        self._response = response
        self.call_count = 0
        self.last_messages: list[Message] = []

    def complete(self, messages: list[Message]) -> LLMResponse:
        """固定レスポンスを返す。呼び出し記録も保持する。

        Args:
            messages: 入力メッセージリスト（記録のみ）。

        Returns:
            固定テキストを含むLLMResponse。
        """
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(
            content=self._response,
            model="mock",
            finish_reason="stop",
        )
