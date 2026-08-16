# Resume Bullets

## 中文版

- 基于 Microsoft Qlib 构建量化研究验证管线，打通 Alpha158 特征、动态因子信号、Top50 组合与含涨跌停/交易成本的执行回测；处理 47.9 万行 CSI300 面板数据。
- 识别并修复 Alpha158 `T+1 → T+2` 标签的一期滞后前视问题，按交易日实现 train/validation/test purge，并以 11 项测试和 86.27% 核心覆盖率固化研究边界。
- 将 Rank IC、ICIR、换手、成本后超额和最大回撤转化为研究晋级 Gate；在扩展信号未通过成本后验证时输出 BLOCK，避免将统计相关性包装成可交易策略。

## English

- Built a Qlib-based quant research validation pipeline covering Alpha158 features, dynamic factor signals, Top-50 portfolio construction, and execution backtesting with price-limit and transaction-cost constraints on 479K CSI300 panel rows.
- Eliminated a label-availability leakage issue for Qlib's T+1-to-T+2 target using trading-calendar purging across train, validation, and test boundaries; enforced the controls with 11 tests and 86.27% core-module coverage.
- Implemented a research-promotion gate using Rank IC, turnover, after-cost excess return, information ratio, and drawdown; blocked extension signals that failed execution-level validation instead of presenting statistical correlation as a tradable strategy.
