"""LLMクライアントの抽象インターフェース。

ローカルLLM・クラウドAPI・モックを同一インターフェースで扱うための基底クラス。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """チャット形式のメッセージ。"""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMResponse:
    """LLMからのレスポンス。"""

    content: str
    model: str
    finish_reason: str = "stop"


class BaseLLMClient(ABC):
    """LLMクライアントの抽象基底クラス。

    ローカルLLM（Ollama等）やクラウドAPI（OpenAI等）は
    このクラスを継承して実装する。
    """

    @abstractmethod
    def complete(self, messages: list[Message]) -> LLMResponse:
        """メッセージリストを受け取りLLMのレスポンスを返す。

        Args:
            messages: チャット履歴を含むメッセージのリスト。

        Returns:
            LLMからのレスポンス。
        """
        ...

    def chat(self, user_input: str, system_prompt: str | None = None) -> str:
        """シンプルな1ターンのチャットインターフェース。

        Args:
            user_input: ユーザーからの入力。
            system_prompt: システムプロンプト（省略可）。

        Returns:
            LLMの返答テキスト。
        """
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_input))
        response = self.complete(messages)
        return response.content
