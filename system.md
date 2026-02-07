# Financial Agent - System Design

## High-Level Architecture

```mermaid
graph TB
    subgraph Entry["Entry Points"]
        M[main.py<br>CLI Runner]
        E[eval/runner.py<br>Eval Runner]
    end
    
    subgraph Core["Core Agent (src/)"]
        G[graph.py<br>LangGraph Workflow]
        S[state.py<br>AgentState]
        T[tools.py<br>LangChain Tools]
        C[client.py<br>FMPClient]
        U[utils.py<br>MarkdownLogger]
    end
    
    subgraph Eval["Evaluation System (eval/)"]
        DS[datasets.py<br>Test Cases]
        EV[evaluators.py<br>Custom Evaluators]
        CF[conftest.py<br>Pytest Fixtures]
        TE[test_eval.py<br>Unit Tests]
    end
    
    subgraph External["External Services"]
        FMP[FMP API<br>Financial Data]
        LS[LangSmith<br>Tracing & Evals]
        OAI[OpenAI<br>GPT-5-mini]
    end
    
    M --> G
    E --> G
    E --> DS
    E --> EV
    G --> S
    G --> T
    T --> C
    C --> FMP
    G --> OAI
    EV --> OAI
    E --> LS
    M --> U
```

---

## LangGraph Workflow

```mermaid
stateDiagram-v2
    [*] --> Researcher: START
    Researcher --> Tools: tool_calls exist
    Researcher --> Writer: no tool_calls
    Tools --> Writer
    Writer --> [*]: END
```

| Node | Role | Model |
|------|------|-------|
| **Researcher** | Decide which tools to call | GPT-5-mini (with tools) |
| **Tools** | Execute tool calls | ToolNode (prebuilt) |
| **Writer** | Generate final report | GPT-5-mini (no tools) |

---

## Component Breakdown

### 1. Core Agent (`src/`)

| File | Purpose |
|------|---------|
| `graph.py` | LangGraph workflow definition (nodes, edges, conditions) |
| `state.py` | `AgentState` - holds `messages` + `ticker` |
| `tools.py` | LangChain tools wrapping FMP API calls |
| `client.py` | Async HTTP client for FMP API |
| `utils.py` | `MarkdownLogger` for saving reports |

### 2. Tools Layer

```mermaid
graph LR
    LLM[LLM with tools] --> |tool_calls| T1[get_company_profile]
    LLM --> |tool_calls| T2[get_financial_ratios]
    LLM --> |tool_calls| T3[get_financial_statements]
    
    T1 --> C[FMPClient]
    T2 --> C
    T3 --> C
    
    C --> |HTTP| FMP[FMP API]
```

| Tool | Endpoint | Returns |
|------|----------|---------|
| `get_company_profile` | `/profile`, `/quote` | Company info, sector, CEO, market cap |
| `get_financial_ratios` | `/ratios-ttm`, `/key-metrics-ttm` | PE, EPS, ROE, Debt/Equity |
| `get_financial_statements` | `/income-statement`, `/balance-sheet`, `/cash-flow` | 4Y historical financials |

### 3. Evaluation System (`eval/`)

```mermaid
graph LR
    DS[datasets.py<br>TEST_CASES] --> R[runner.py]
    R --> |run_agent_sync| G[graph.py]
    G --> |outputs| EV[evaluators.py]
    
    EV --> F[Faithfulness<br>LLM-as-Judge]
    EV --> C[Completeness<br>Rule-based]
    EV --> TC[Tool Coverage<br>Rule-based]
    
    R --> LS[LangSmith<br>Dashboard]
```

| Evaluator | Method | Checks |
|-----------|--------|--------|
| **Faithfulness** | LLM-as-Judge (GPT-5-mini) | No hallucination, data from tools only |
| **Completeness** | Rule-based | Required sections present |
| **Tool Coverage** | Rule-based | All expected tools called |

---

## Data Flow

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant G as Graph
    participant R as Researcher
    participant T as Tools
    participant FMP as FMP API
    participant W as Writer
    participant L as Logger
    
    U->>G: invoke(ticker="AAPL")
    G->>R: state{messages, ticker}
    R->>R: Decide tool calls
    R->>T: tool_calls[3]
    
    par Parallel API Calls
        T->>FMP: get_profile
        T->>FMP: get_ratios
        T->>FMP: get_statements
    end
    
    FMP-->>T: Financial data
    T->>W: Tool outputs in messages
    W->>W: Generate report
    W-->>G: Final AIMessage
    G-->>U: Report markdown
    U->>L: Save to logs/
```

---

## Key Design Patterns

| Pattern | Usage |
|---------|-------|
| **State Machine** | LangGraph nodes + conditional edges |
| **Dependency Injection** | Tools bound to LLM at setup |
| **EAFP** | Try API first, handle errors after |
| **Async/Await** | All API calls async, `asyncio.gather` for parallelism |
| **Pydantic Models** | Type-safe API responses |

---

## Directory Structure

```
financial-agent/
├── main.py              # CLI entry point
├── src/
│   ├── graph.py         # LangGraph workflow
│   ├── state.py         # AgentState definition
│   ├── tools.py         # LangChain tools
│   ├── client.py        # FMP API client
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # MarkdownLogger
├── eval/
│   ├── runner.py        # Eval orchestrator
│   ├── datasets.py      # Test cases + LangSmith dataset
│   ├── evaluators.py    # Custom evaluators
│   ├── conftest.py      # Pytest fixtures
│   └── test_eval.py     # Unit tests
└── logs/                # Generated reports
```
