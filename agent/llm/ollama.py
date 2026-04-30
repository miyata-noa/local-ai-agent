"""Ollamaを使ったローカルLLMクライアント。

Ollamaが起動済みであれば、クラウドAPIなしで動作する。
DGX Spark等のローカルGPU環境での運用を想定。

使用前提:
    - Ollamaがインストール済みで起動していること
    - 使用するモデルがpull済みであること
      例: ollama pull qwen2.5:7b

参考:
    OpenClawのOllama統合ではnative API (/api/chat) を使用する。
    /v1 (OpenAI互換) はtool callingが壊れるため使用しない。
"""
import httpx

from agent.llm.base import BaseLLMClient, LLMResponse, Message


class OllamaClient(BaseLLMClient):
    """OllamaのネイティブAPIを使ったローカルLLMクライアント。

    OpenClawのOllama統合仕様に合わせ、/api/chat エンドポイントを使用する。
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        """初期化。

        Args:
            model: 使用するOllamaモデル名。
            base_url: OllamaサーバーのURL。デフォルトはローカル。
            timeout: リクエストタイムアウト秒数。
                     ローカルLLMは初回推論が遅いため長めに設定。
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete(self, messages: list[Message]) -> LLMResponse:
        """OllamaのネイティブAPIでLLM推論を実行する。

        Args:
            messages: チャット履歴を含むメッセージのリスト。

        Returns:
            LLMからのレスポンス。

        Raises:
            httpx.HTTPError: Ollamaサーバーへの接続に失敗した場合。
            RuntimeError: レスポンスのパースに失敗した場合。
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": False,
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()

        data = response.json()

        try:
            content: str = data["message"]["content"]
            finish_reason: str = data.get("done_reason", "stop")
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected Ollama response format: {data}"
            ) from e

        return LLMResponse(
            content=content,
            model=self._model,
            finish_reason=finish_reason,
        )

    @property
    def model(self) -> str:
        """使用中のモデル名。"""
        return self._model
