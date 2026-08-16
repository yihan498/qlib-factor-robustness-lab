"""Publish the v5 ownership-first narrative without rewriting historical v4 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"


REPLACEMENTS = {
    "v4 · 2026-08-16": "v5 · 2026-08-16",
    "本研究检验基于历史信息系数的稳定性加权，能否改善 Alpha158 截面因子组合在交易约束下的样本外表现。扩展方案在测试期取得正 Rank IC，但未形成正的成本后超额收益，也未优于两个预设对照。":
        "本项目以 Qlib 作为标准化数据、特征和执行基础设施，独立构建标签时序控制、稳定性因子权重、冻结样本外评估与机器研究 Gate。稳定性方案在测试期取得正 Rank IC，但未形成正的成本后超额收益，也未优于两个预设对照。",
    "项目包含两条独立证据链。上游复现验证本地 Qlib 工作流；扩展评估检验独立研究假设。两条结果不构成同一模型的前后比较。":
        "项目研究主线是独立实现的稳定性加权与研究治理系统；Qlib 标准参考工作流只用于标定数据、特征和执行基础设施。项目未修改 Qlib 上游源码，两类结果不构成同一模型的前后比较。",
    "<tr><td>Alpha158 + LightGBM 上游复现</td><td>验证框架、数据与执行环境</td><td>0.0496</td><td>9.56%</td><td>环境可运行</td></tr><tr><td>稳定性加权扩展</td><td>检验权重方法与研究治理</td><td>0.0122</td><td>-17.50%</td><td>未晋级</td></tr>":
        "<tr><td>项目研究主线：稳定性加权</td><td>检验时序、权重方法与研究治理</td><td>0.0122</td><td>-17.50%</td><td>未晋级</td></tr><tr><td>基础设施标定：标准参考工作流</td><td>确认数据、特征与执行环境</td><td>0.0496</td><td>9.56%</td><td>环境通过标定</td></tr>",
}


def main() -> None:
    source = (DOCS / "v4" / "index.html").read_text(encoding="utf-8")
    page = source
    for old, new in REPLACEMENTS.items():
        if old not in page:
            raise RuntimeError(f"v5 narrative source text not found: {old[:60]}")
        page = page.replace(old, new)

    target = DOCS / "v5"
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(page, encoding="utf-8")
    (DOCS / "index.html").write_text(
        page.replace('src="../assets/', 'src="assets/'), encoding="utf-8"
    )
    manifest = {
        "showcase_version": "v5",
        "format": "institutional research note",
        "narrative": "ownership-first; Qlib identified as infrastructure",
        "gate_decision": "BLOCK",
        "outputs": ["docs/index.html", "docs/v5/index.html"],
        "historical_version_preserved": "docs/v4/index.html",
    }
    (EVIDENCE / "v5_showcase_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
