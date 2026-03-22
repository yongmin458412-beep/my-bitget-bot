"""Prompt builders for OpenAI assistant tasks."""

from __future__ import annotations

from textwrap import dedent


def news_system_prompt() -> str:
    """System instructions for news analysis."""

    return dedent(
        """
        너는 선물 자동매매 시스템의 뉴스 필터 보조 엔진이다.
        목표는 과장 없는 리스크 평가다.
        절대 수익을 보장하거나 매수/매도 단독 결정을 내리지 않는다.
        정보가 모호하면 direction_bias는 uncertain, should_block_new_entries는 보수적으로 판단하라.
        JSON 스키마를 반드시 지켜라.
        """
    ).strip()


def news_user_prompt(title: str, body: str, source: str, published_at: str, related_assets: list[str]) -> str:
    """User prompt for structured news analysis."""

    return dedent(
        f"""
        아래 뉴스/이벤트를 분석하라.

        제목: {title}
        본문: {body}
        출처: {source}
        시각: {published_at}
        추정 관련 자산: {", ".join(related_assets) if related_assets else "없음"}

        요구사항:
        - summary_ko는 한국어 2~3문장.
        - 과장 표현 금지.
        - 불확실하면 uncertain.
        - 신규 진입 차단 필요 여부를 보수적으로 판단.
        """
    ).strip()


def why_prompt(symbol: str, details: str) -> str:
    """Prompt for explaining why a symbol was or was not traded."""

    return dedent(
        f"""
        사용자가 {symbol}의 매매 이유를 물었다.
        아래 구조화 정보를 바탕으로 한국어로 짧고 명확하게 설명하라.
        거래 판단은 이미 규칙 엔진이 했고, 너는 설명 보조만 담당한다.

        데이터:
        {details}
        """
    ).strip()


def journal_prompt(serialized_trades: str) -> str:
    """Prompt for summarizing the trade journal."""

    return dedent(
        f"""
        다음 거래 일지를 요약하라.
        반드시 한국어로 작성하고, 추상적 격려보다 실제 개선 포인트를 우선한다.

        데이터:
        {serialized_trades}
        """
    ).strip()

