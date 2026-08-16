# Upstream Ledger

## Microsoft Qlib

- Repository: https://github.com/microsoft/qlib
- License: MIT
- Intended use: data format, Alpha158 handler, official LightGBM workflow, backtest and portfolio analysis infrastructure
- Executed Qlib release: `v0.9.7`
- Executed release commit: `da920b7f954f48ab1bb64117c976710de198373e`
- Installed runtime: `pyqlib==0.9.7`, `lightgbm==4.7.0`
- Separately cloned reference snapshot: `79633dd9506ea689e5400dea0197717b5b3d74b7` (2026-07-23). This snapshot was inspected for current documentation but was **not** the executed runtime.
- Local reference path: `upstream/qlib/`（Git 忽略，不作为个人源码提交）
- Project modifications to upstream source: none
- Attribution rule: README、研究报告和未来 GitHub 页面均保留 Qlib 名称、链接和 MIT notice；不得使用 Microsoft/Qlib 品牌暗示官方认可

## Prior open-source adaptation lessons reused

- 先记录精确提交和许可证，再实施；
- 上游参考与原创源码分离；
- 先用标准参考工作流标定研究基础设施，再独立实施单一可检验研究假设；
- 把研究原则变成自动测试和机器可读结果；
- 最终结论以真实最后一跳实验为准，不以代码存在或 notebook 打开为准。

## Data provenance

- Download entry point: `qlib.tests.data.GetData` bundled with pyqlib 0.9.7.
- Remote release host used by that downloader: `SunsetWolf/qlib_dataset`.
- Downloader warning: the example data is collected from Yahoo Finance and may have imperfect quality.
- Repository policy: call it **community-maintained Qlib example data**, not official or institutional-grade market data.
