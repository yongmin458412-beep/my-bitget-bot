# Bitget 선물 자동매매 시스템

Bitget 공식 선물 REST/WebSocket API를 기준으로 설계한 멀티파일 Python 프로젝트입니다.  
기본 모드는 반드시 `DEMO`이며, `LIVE` 전환은 `.env`, `bot_settings.json`, Streamlit 최종 확인이 모두 맞아야만 허용됩니다.  
전략 엔진, 리스크 엔진, 실행 엔진, 뉴스 엔진, Telegram 엔진, Streamlit 제어판을 분리해 24시간 운영과 장애 복구를 우선했습니다.

## 핵심 특징

- Bitget 선물 전체 심볼 수집 후 활성 유니버스 재선정
- Break & Retest + Liquidity Raid 규칙형 전략
- EV 필터, 수수료/슬리피지/뉴스/펀딩 리스크 반영
- 구조형 손절 + ATR buffer + TP1 후 BE 이동
- SQLite 기반 상태 복원, 주문/포지션/일지 저장
- Telegram 실시간 알림 및 조회 명령어
- Streamlit 기반 제어판
- OpenAI Responses API + Structured Outputs 기반 뉴스 분석
- 동일 전략 로직 재사용 백테스트 엔진

## 프로젝트 구조

```text
my-bitget-bot/
  README.md
  requirements.txt
  .env.example
  docker-compose.yml
  Dockerfile
  run_bot.py
  run_streamlit.py
  config/
    defaults.json
    bot_settings.json
  app/
    __init__.py
    main.py
  core/
    __init__.py
    settings.py
    logger.py
    utils.py
    time_utils.py
    persistence.py
    state_store.py
    enums.py
  exchange/
    __init__.py
    bitget_rest.py
    bitget_ws.py
    bitget_models.py
    bitget_demo.py
    bitget_live.py
  market/
    __init__.py
    universe.py
    symbol_ranker.py
    klines.py
    orderbook.py
    indicators.py
    sessions.py
    market_regime.py
    sr_levels.py
    volume_profile.py
  strategy/
    __init__.py
    base.py
    break_retest.py
    liquidity_raid.py
    confirmation.py
    signal_score.py
    ev_filter.py
  risk/
    __init__.py
    position_sizing.py
    risk_engine.py
    stops.py
    trade_guard.py
  execution/
    __init__.py
    router.py
    order_manager.py
    sltp_manager.py
    fill_handler.py
  news/
    __init__.py
    collector.py
    rss_sources.py
    economic_calendar.py
    parser.py
    analyzer.py
    impact_filter.py
  ai/
    __init__.py
    client.py
    schemas.py
    prompts.py
    summarizer.py
  telegram_bot/
    __init__.py
    bot.py
    formatters.py
    commands.py
    keyboards.py
  dashboard/
    __init__.py
    common.py
    streamlit_app.py
    pages/
      1_overview.py
      2_positions.py
      3_symbols.py
      4_news.py
      5_journal.py
      6_settings.py
  journal/
    __init__.py
    trade_journal.py
    performance.py
  backtest/
    __init__.py
    engine.py
    simulator.py
    reports.py
  tests/
    test_settings.py
    test_indicators.py
    test_signal_score.py
    test_risk_engine.py
    test_telegram_format.py
```

## 설치

작업 디렉터리: `my-bitget-bot`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

예상 산출물
- `.venv/`
- `.env`
- 설치된 Python 패키지

## 환경변수 설정 위치

위치: `my-bitget-bot/.env`

- Bitget 데모 API 키
  - `BITGET_DEMO_API_KEY`
  - `BITGET_DEMO_API_SECRET`
  - `BITGET_DEMO_API_PASSPHRASE`
- Bitget 실전 API 키
  - `BITGET_API_KEY`
  - `BITGET_API_SECRET`
  - `BITGET_API_PASSPHRASE`
- OpenAI
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- Telegram
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- LIVE 안전장치
  - `LIVE_TRADING_ENABLED=false`

## 설정 파일 위치

- 기본값: `config/defaults.json`
- 런타임/UI 수정값: `config/bot_settings.json`

`/status` 응답과 Streamlit은 이 두 파일과 현재 런타임 상태를 함께 사용합니다.

## 데모 모드 실행 방법

작업 디렉터리: `my-bitget-bot`

```bash
python run_bot.py
```

전제조건
- `.env` 설정 완료
- `config/bot_settings.json`의 `mode`가 `DEMO`

예상 산출물
- `logs/bot.log`
- `state/runtime_state.json`
- `data/trading_bot.sqlite3`

## Streamlit 실행 방법

작업 디렉터리: `my-bitget-bot`

```bash
python run_streamlit.py
```

브라우저 접속
- `http://localhost:8501`

## 실전 전환 방법

실전 전환은 아래 4개를 모두 만족해야 합니다.

1. `.env`에 `LIVE_TRADING_ENABLED=true`
2. `config/bot_settings.json` 또는 Streamlit Settings에서 `mode=LIVE`
3. Streamlit Settings에서 `LIVE 최종 확인` 체크
4. `/reload` 또는 봇 재시작

안전 장치
- 기본값은 항상 `DEMO`
- `LIVE` 검증 실패 시 시작 단계에서 예외 발생
- Telegram에 모드 변경/재시작 알림 전송

## Telegram 봇 생성/연결 방법

1. Telegram에서 `@BotFather` 실행
2. `/newbot`으로 봇 생성
3. 발급된 토큰을 `.env`의 `TELEGRAM_BOT_TOKEN`에 저장
4. 본인 또는 운영 채팅방 ID를 `TELEGRAM_CHAT_ID`에 저장
5. `config/bot_settings.json`의 `telegram.enabled=true`
6. 필요 시 `telegram.admin_ids`에 허용 사용자 ID 등록

지원 명령
- `/start`
- `/help`
- `/status`
- `/positions`
- `/balance`
- `/pnl`
- `/watchlist`
- `/signals`
- `/today`
- `/journal`
- `/why SYMBOL`
- `/pause`
- `/resume`
- `/mode`
- `/risk`
- `/settings`
- `/close SYMBOL`
- `/closeall`
- `/news`
- `/events`
- `/reload`

참고
- 현재 저장소 규칙 때문에 Telegram은 코어 필수 의존성이 아니라 선택형 엔진입니다.
- `TELEGRAM_BOT_TOKEN`이 비어 있으면 나머지 시스템은 계속 동작합니다.

## 백테스트 실행 방법

작업 디렉터리: `my-bitget-bot`

입력 CSV 형식 예시
- 컬럼: `timestamp,open,high,low,close,volume[,quote_volume]`
- 타임프레임: 1분봉

실행 명령

```bash
python -m backtest.engine --data data/backtest/BTCUSDT_1m.csv --symbol BTCUSDT --output data/backtest/report.md
```

예상 산출물
- `data/backtest/report.md`

## Docker 실행

```bash
docker compose up --build
```

대시보드
- `http://localhost:8501`

## 테스트 실행 방법

작업 디렉터리: `my-bitget-bot`

```bash
pytest -q
```

## 운영 주의사항

- AI는 뉴스 요약/설명/장세 설명 보조만 수행하며, 단독으로 주문하지 않습니다.
- 실거래 전에는 반드시 데모 모드에서 장기간 검증하세요.
- 거래소 API 스펙은 변경될 수 있으므로 배포 전 공식 문서를 다시 확인하세요.
- WebSocket/REST 재시도 로직이 있어도 네트워크 장애 중 손절 보장은 거래소/네트워크 상태에 영향을 받습니다.
- 일부 경제 일정 소스는 공식 RSS/페이지에 의존하므로 시점별 형식 변경 가능성이 있습니다.
- 현재 백테스트 엔진은 CSV 기반 단일 심볼 루프이며, 초고빈도 체결 시뮬레이션까지는 하지 않습니다.

## 권장 운영 순서

1. `.env` 설정
2. `config/bot_settings.json` 검토
3. `pytest -q`
4. `python run_bot.py`
5. `python run_streamlit.py`
6. Telegram `/status` 확인
