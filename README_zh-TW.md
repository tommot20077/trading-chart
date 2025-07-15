# TradingChart - 交易圖表系統

現代化的交易資料基礎設施，採用乾淨架構設計。

## 🎯 專案概述

- **目的**：具備乾淨架構的現代化交易資料基礎設施
- **語言**：Python 3.12+
- **架構**：四層 monorepo 架構，具備相依性注入
- **套件管理器**：uv（嚴格執行）
- **測試框架**：pytest 搭配多層測試策略
- **程式碼品質**：ruff（程式碼檢查）、mypy（型別檢查）、自動化品質檢查

## 🏗️ 架構層級

### Core 層 (`src/core/`)
- **目的**：定義合約並提供預設的記憶體內實作
- **原則**：最小化、基礎相依性
- **包含**：介面、模型、基礎實作
- **相依性**：無（基礎層）

### Libs 層 (`src/libs/`)
- **目的**：可重複使用的內部商業邏輯
- **原則**：內部、商業導向功能
- **包含**：驗證、工具、商業服務
- **相依性**：僅 core

### Integrations 層 (`src/integrations/`)
- **目的**：外部基礎設施互動
- **原則**：外部、基礎設施導向
- **包含**：資料庫適配器、外部 API 客戶端、訊息傳遞
- **相依性**：僅 core

### Apps 層 (`src/apps/`)
- **目的**：應用程式組裝和特定商業邏輯
- **原則**：組裝與特定邏輯
- **包含**：FastAPI 應用程式、CLI 工具、工作程序
- **相依性**：core、libs、integrations

## 🚀 開發指令

### 套件管理
- **必須**：所有套件管理操作都使用 `uv`
- `uv add <dependency>`：新增相依性到特定套件
- `uv sync`：同步工作空間相依性
- `uv run <command>`：在虛擬環境中執行指令

### 測試策略
- **多層測試**：單元 → 整合 → 合約 → 端對端 → 效能
- **測試標記**：使用 `@pytest.mark.unit`、`@pytest.mark.integration`、`@pytest.mark.benchmark` 等
- **時間控制**：直接使用 `time_machine`（不使用自訂包裝器）
- **覆蓋率**：要求最低 80% 覆蓋率

### 品質保證
- **自動化檢查**：所有提交都必須通過品質檢查
- **工具**：ruff（程式碼檢查）、mypy（型別檢查）、pytest（測試）
- **預提交鉤子**：自動執行標準
- **不可繞過**：提交時絕不使用 `--no-verify`

### 目前效能基準
- **單例存取**：~77ns（每秒 1300 萬次操作）
- **基礎驗證器**：~400μs（每秒 2500 次操作）
- **TIMEZONE 驗證**：~400μs（標準）/ ~10.8ms（大小寫標準化）
- **完整物件建立**：~450μs（每秒 2200 次操作）

## 🔧 程式碼標準

### 檔案結構
- **ABOUTME 註解**：所有檔案都必須以 2 行 ABOUTME 註解開始
  ```python
  # ABOUTME: 此檔案功能的簡短描述
  # ABOUTME: 額外的背景或目的
  ```
- **介面分離**：遵循 ISP 原則以獲得乾淨的合約
- **相依性注入**：使用 dependency-injector 進行 IoC

### 開發原則
- **乾淨架構**：尊重層級邊界和相依性規則
- **SOLID 原則**：特別是單一職責和介面分離
- **測試驅動開發**：先寫測試，再實作
- **基於證據的決策**：提供測試結果和文件
- **漸進式變更**：進行最小合理的變更

## 🛠️ 配置系統

### 強化的驗證器
```python
from core.config._base import BaseCoreSettings

# 這些都會正確運作並標準化：
settings = BaseCoreSettings(
    ENV="PROD",              # → "production"
    LOG_LEVEL="debug",       # → "DEBUG"  
    LOG_FORMAT="structured", # → "json"
    TIMEZONE="america/new_york"  # → "America/New_York"
)
```

### 支援的別名
- **ENV**：`dev`/`develop` → `development`，`prod` → `production`，`stage` → `staging`
- **LOG_FORMAT**：`structured` → `json`，`text` → `console`
- **TIMEZONE**：完整的 IANA 時區驗證與大小寫標準化

## 📚 文件結構

### Core 層 (`src/core/`)
- `core/interfaces/`：抽象合約和協定
- `core/models/`：資料模型（market_data、events、base）
- `core/implementations/`：預設記憶體內實作
- `core/config/`：配置管理
- `tests/`：完整測試套件

### 測試目錄
- `tests/unit/`：單元測試
- `tests/integration/`：整合測試
- `tests/benchmark/`：效能測試
- `tests/contract/`：合約測試

## 🎯 快速開始

### 安裝
```bash
# 複製專案
git clone <repo_url> && cd trading-chart

# 安裝所有相依性
uv sync --dev

# 安裝 core 套件為可編輯模式
uv add --editable src/core
```

### 執行測試
```bash
# 執行所有測試
uv run pytest

# 執行特定類型的測試
uv run pytest -m unit
uv run pytest -m benchmark

# 執行程式碼品質檢查
uv run ruff check .
uv run mypy .
```

### 開發工作流程
```bash
# 檢查程式碼格式
uv run ruff format --check .

# 修復程式碼格式
uv run ruff format .

# 執行所有品質檢查
uv run ruff check . && uv run mypy . && uv run pytest
```

## 🔒 安全框架

- **輸入驗證**：所有進入點的完整清理
- **驗證**：基於 JWT 的驗證與角色存取控制
- **稽核軌跡**：完整的安全事件記錄
- **加密**：靜態和傳輸中的敏感資料加密
- **安全測試**：專門的安全測試套件

## 📈 品質保證

- **自動化測試**：多層測試策略，80% 以上覆蓋率
- **程式碼品質檢查**：自動化 ruff、mypy 和 pytest 檢查
- **效能監控**：內建效能追蹤和警報
- **安全掃描**：定期安全漏洞評估
- **文件標準**：完整的內聯和外部文件

## 🤝 貢獻指南

### Git 工作流程
- **分支策略**：功能分支搭配 PR 整合
- **提交標準**：使用清晰、描述性訊息的慣例提交
- **品質檢查**：所有 PR 都必須通過自動化品質檢查
- **程式碼審查**：合併前必須進行同儕審查

### 測試工作流程
1. **先寫測試**：所有新功能採用 TDD 方法
2. **多層測試**：單元 → 整合 → 合約 → 端對端 → 效能
3. **品質檢查**：所有層級都要求 80% 以上覆蓋率
4. **時間控制**：使用 `time_machine` 進行確定性測試

## 📞 支援

- **問題回報**：使用 GitHub Issues
- **功能請求**：透過 GitHub Discussions
- **文件**：查看 `ai-docs/` 目錄
- **範例**：查看 `examples/` 目錄（即將推出）

## 📄 授權

此專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案。

---

**注意**：此專案正在積極開發中。API 可能會變更，直到達到 1.0.0 版本。