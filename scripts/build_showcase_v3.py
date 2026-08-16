"""Build the versioned v3 technical case study for web, print, and GitHub."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "portfolio_v2"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
PDF_DIR = ROOT / "output" / "pdf"
EVIDENCE = ROOT / "evidence"
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")
INK, MUTED, LINE = "#132238", "#5b6b7f", "#d7dee8"
BLUE, CYAN, GOLD, RED, GREEN = "#174a7e", "#0f7490", "#b7791f", "#b42318", "#067647"
METHODS = ["fixed_equal_signed", "rolling_ic_60", "adaptive_shrinkage"]
METHOD_COLORS = dict(zip(METHODS, ["#66758a", "#c88a18", "#1769aa"]))
LABELS = dict(zip(METHODS, ["静态方向等权", "60日 Rolling-IC", "稳定性收缩组合"]))
GATE_LABELS = {
    "test_days": "测试标签日",
    "rank_ic": "Rank IC",
    "rank_icir": "年化 Rank ICIR",
    "after_cost_annualized_excess": "成本后年化超额",
    "after_cost_information_ratio": "成本后信息比率",
    "max_drawdown": "最大回撤下限",
    "return_advantage_vs_best_control": "相对最佳对照收益优势",
    "ir_advantage_vs_best_control": "相对最佳对照 IR 优势",
}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_gate(value: float, key: str) -> str:
    if key in {"after_cost_annualized_excess", "max_drawdown", "return_advantage_vs_best_control"}:
        return pct(value)
    if key == "test_days":
        return str(int(value))
    return f"{value:.3f}"


def load() -> tuple[dict, dict]:
    experiment = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    official = json.loads((EVIDENCE / "official_baseline_metrics.json").read_text(encoding="utf-8"))
    if "research_promotion_gate" not in experiment:
        raise RuntimeError("results.json predates the machine-readable research promotion gate")
    return experiment, official


def setup_dirs() -> None:
    for directory in [ASSETS, PDF_DIR, DOCS / "v2", DOCS / "v3", EVIDENCE]:
        directory.mkdir(parents=True, exist_ok=True)
    if (DOCS / "index.html").exists() and not (DOCS / "v2" / "index.html").exists():
        shutil.copyfile(DOCS / "index.html", DOCS / "v2" / "index.html")


def plot_architecture(experiment: dict, font: FontProperties) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 4.6))
    ax.axis("off")
    layers = [
        ("01 数据与特征", "社区 Qlib 示例数据\nCSI300 历史成分\nAlpha158 特征/标签", BLUE),
        ("02 研究与选择", "训练期因子筛选\n两交易日信息滞后\n验证期 12 个候选", CYAN),
        ("03 组合与执行", "三类截面信号\nTop50 / Drop5\n涨跌停与双边成本", GOLD),
        ("04 治理与证据", "冻结测试期\n机器可读 Gate\n测试/哈希/版本留痕", RED),
    ]
    for i, (title, body, color) in enumerate(layers):
        x = 0.03 + i * 0.245
        ax.add_patch(plt.Rectangle((x, 0.27), 0.205, 0.56, facecolor="white", edgecolor=color,
                                   linewidth=2, transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((x, 0.70), 0.205, 0.13, facecolor=color, edgecolor=color,
                                   transform=ax.transAxes))
        ax.text(x + 0.015, 0.765, title, color="white", fontsize=11, fontproperties=font,
                va="center", transform=ax.transAxes)
        ax.text(x + 0.1025, 0.485, body, color=INK, fontsize=10, fontproperties=font,
                ha="center", va="center", linespacing=1.6, transform=ax.transAxes)
        if i < 3:
            ax.annotate("", xy=(x + 0.235, 0.55), xytext=(x + 0.208, 0.55),
                        xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.5})
    manifest = experiment["data_manifest"]
    ax.text(0.03, 0.12,
            f"证据规模  {manifest['feature_rows']:,} 行 × {manifest['instruments']} 个历史成分证券  |  "
            f"Python {experiment['runtime']['python']} / pyqlib {experiment['runtime']['pyqlib']} / LightGBM {experiment['runtime']['lightgbm']}",
            color=MUTED, fontsize=9.5, fontproperties=font, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(ASSETS / "v3_architecture.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_temporal_protocol(experiment: dict, font: FontProperties) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 4.2))
    ax.axis("off")
    segments = [
        (0.05, 0.34, "训练 / 因子筛选", "2014-01 — 2015-12", BLUE),
        (0.39, 0.18, "验证 / 参数选择", "2016-01 — 2016-12", CYAN),
        (0.57, 0.38, "冻结测试 / 最终评估", "2017-01 — 2020-07", GOLD),
    ]
    for x, width, title, dates, color in segments:
        ax.add_patch(plt.Rectangle((x, 0.59), width, 0.20, facecolor=color, edgecolor="none",
                                   transform=ax.transAxes))
        ax.text(x + 0.015, 0.715, title, color="white", fontproperties=font, fontsize=11,
                transform=ax.transAxes)
        ax.text(x + 0.015, 0.625, dates, color="white", fontproperties=font, fontsize=9,
                transform=ax.transAxes)
    tc = experiment["temporal_controls"]
    ax.text(0.39, 0.52, f"有效 IC 截止 {tc['train_effective_ic_cutoff']}", color=MUTED,
            fontproperties=font, fontsize=9, ha="center", transform=ax.transAxes)
    ax.text(0.57, 0.52, f"有效 IC 截止 {tc['validation_effective_ic_cutoff']}", color=MUTED,
            fontproperties=font, fontsize=9, ha="center", transform=ax.transAxes)
    positions = [0.12, 0.28, 0.44, 0.60, 0.76]
    labels = ["T-2", "T-1", "T / 形成信号", "T+1", "T+2"]
    notes = ["最近可用 IC", "标签尚未完整", "仅用已知信息", "标签起点", "标签终点"]
    ax.plot([positions[0], positions[-1]], [0.31, 0.31], color=LINE, lw=2,
            transform=ax.transAxes, zorder=1)
    for x, label, note in zip(positions, labels, notes):
        color = GREEN if label == "T-2" else (RED if "+" in label else BLUE)
        ax.scatter([x], [0.31], s=75, color=color, transform=ax.transAxes, zorder=3)
        ax.text(x, 0.23, label, ha="center", fontproperties=font, fontsize=9.5,
                color=INK, transform=ax.transAxes)
        ax.text(x, 0.15, note, ha="center", fontproperties=font, fontsize=8.2,
                color=MUTED, transform=ax.transAxes)
    ax.text(0.95, 0.19, "标签定义: T+1 → T+2 收益\n信息滞后: 2 个交易日", ha="right",
            fontproperties=font, fontsize=9.5, color=RED, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(ASSETS / "v3_temporal_protocol.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_validation_surface(experiment: dict, font: FontProperties) -> None:
    candidates = pd.DataFrame(experiment["validation_candidates"])
    candidates["row"] = candidates.apply(
        lambda row: f"W={int(row['window'])}  λ={row['uncertainty_penalty']:.1f}", axis=1)
    rows = ["W=60  λ=0.5", "W=60  λ=1.0", "W=120  λ=0.5", "W=120  λ=1.0"]
    cols = sorted(candidates["shrinkage_to_dynamic"].unique())
    matrix = candidates.pivot(index="row", columns="shrinkage_to_dynamic",
                              values="rank_icir_annualized").reindex(rows)[cols]
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    image_map = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)), [f"动态权重占比 {c:.0%}" for c in cols],
                  fontproperties=font, fontsize=10)
    ax.set_yticks(range(len(rows)), rows, fontproperties=font, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix.iloc[i, j] > 4.72 else INK, fontsize=10)
    selected = experiment["selected_adaptive_parameters"]
    selected_row = rows.index(f"W={int(selected['window'])}  λ={selected['uncertainty_penalty']:.1f}")
    selected_col = cols.index(selected["shrinkage_to_dynamic"])
    ax.add_patch(plt.Rectangle((selected_col - 0.49, selected_row - 0.49), 0.98, 0.98,
                               fill=False, edgecolor=RED, linewidth=3))
    ax.set_title("验证期候选面：按年化 Rank ICIR 选择一次参数", fontproperties=font,
                 fontsize=14, loc="left", color=INK)
    ax.text(0, -0.22, "红框为冻结到测试期的配置；测试期不再调参。", transform=ax.transAxes,
            fontproperties=font, fontsize=9, color=MUTED)
    fig.colorbar(image_map, ax=ax, fraction=0.025, pad=0.03, label="Validation Rank ICIR")
    fig.tight_layout()
    fig.savefig(ASSETS / "v3_validation_surface.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_execution_diagnostics(experiment: dict, font: FontProperties) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2))
    ax_curve, ax_cost, ax_dd, ax_scatter = axes.flat
    for name in METHODS:
        report = pd.read_csv(ARTIFACTS / f"backtest_{name}.csv", index_col=0, parse_dates=True)
        cumulative = (report["return"] - report["bench"] - report["cost"]).cumsum()
        drawdown = cumulative - cumulative.cummax()
        ax_curve.plot(cumulative.index, cumulative, color=METHOD_COLORS[name], label=LABELS[name], lw=1.9)
        ax_dd.plot(drawdown.index, drawdown, color=METHOD_COLORS[name], label=LABELS[name], lw=1.6)
    for ax, title, ylabel in [(ax_curve, "A. 成本后累计超额路径", "累计超额"),
                              (ax_dd, "C. 超额收益回撤路径", "回撤")]:
        ax.axhline(0, color=LINE, lw=0.9)
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left", color=INK)
        ax.set_ylabel(ylabel, fontproperties=font)
        ax.grid(alpha=0.18)
    ax_curve.legend(prop=font, frameon=False, ncol=3, fontsize=8)
    x = np.arange(len(METHODS))
    before = [experiment["test_results"][n]["portfolio"]["annualized_excess_return_before_cost"] for n in METHODS]
    after = [experiment["test_results"][n]["portfolio"]["annualized_excess_return_after_cost"] for n in METHODS]
    width = 0.34
    ax_cost.bar(x - width / 2, before, width, label="成本前", color="#aab6c5")
    ax_cost.bar(x + width / 2, after, width, label="成本后", color=[METHOD_COLORS[n] for n in METHODS])
    ax_cost.axhline(0, color=INK, lw=0.8)
    ax_cost.set_xticks(x, [LABELS[n] for n in METHODS], fontproperties=font, fontsize=8, rotation=12)
    ax_cost.set_title("B. 执行成本侵蚀", fontproperties=font, fontsize=12, loc="left", color=INK)
    ax_cost.legend(prop=font, frameon=False)
    ax_cost.grid(axis="y", alpha=0.18)
    for name in METHODS:
        result = experiment["test_results"][name]
        xval = result["signal"]["rank_ic_mean"]
        yval = result["portfolio"]["annualized_excess_return_after_cost"]
        ax_scatter.scatter(xval, yval, s=85, color=METHOD_COLORS[name], edgecolor="white", linewidth=0.8)
        ax_scatter.annotate(LABELS[name], (xval, yval), xytext=(6, 6), textcoords="offset points",
                            fontproperties=font, fontsize=8)
    ax_scatter.axhline(0, color=RED, lw=0.9, ls="--")
    ax_scatter.set_title("D. 信号质量未转化为可交易收益", fontproperties=font,
                         fontsize=12, loc="left", color=INK)
    ax_scatter.set_xlabel("测试期 Rank IC", fontproperties=font)
    ax_scatter.set_ylabel("成本后年化超额", fontproperties=font)
    ax_scatter.grid(alpha=0.18)
    fig.tight_layout(pad=2)
    fig.savefig(ASSETS / "v3_execution_diagnostics.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gate_rows(experiment: dict) -> list[list[str]]:
    rows = [["检查项", "观测值", "阈值", "结果"]]
    for key, detail in experiment["research_promotion_gate"]["checks"].items():
        rows.append([GATE_LABELS[key], format_gate(detail["observed"], key),
                     format_gate(detail["threshold"], key), "PASS" if detail["passed"] else "FAIL"])
    return rows


def result_rows(experiment: dict) -> list[list[str]]:
    rows = [["扩展方案", "Rank IC", "ICIR", "成本前", "成本后", "IR", "最大回撤", "换手"]]
    for name in METHODS:
        item = experiment["test_results"][name]
        rows.append([LABELS[name], f"{item['signal']['rank_ic_mean']:.4f}",
                     f"{item['signal']['rank_icir_annualized']:.3f}",
                     pct(item["portfolio"]["annualized_excess_return_before_cost"]),
                     pct(item["portfolio"]["annualized_excess_return_after_cost"]),
                     f"{item['portfolio']['information_ratio_after_cost']:.3f}",
                     pct(item["portfolio"]["max_drawdown_after_cost"]),
                     f"{item['portfolio']['average_turnover']:.3f}"])
    return rows


def build_html(experiment: dict, official: dict) -> None:
    gate = experiment["research_promotion_gate"]
    check_html = "".join(
        f"<tr><td>{GATE_LABELS[key]}</td><td>{format_gate(d['observed'], key)}</td>"
        f"<td>{format_gate(d['threshold'], key)}</td><td><span class='pill {'pass' if d['passed'] else 'fail'}'>"
        f"{'PASS' if d['passed'] else 'FAIL'}</span></td></tr>"
        for key, d in gate["checks"].items()
    )
    result_html = "".join(
        f"<tr><td>{LABELS[n]}</td><td>{experiment['test_results'][n]['signal']['rank_ic_mean']:.4f}</td>"
        f"<td>{experiment['test_results'][n]['signal']['rank_icir_annualized']:.3f}</td>"
        f"<td>{pct(experiment['test_results'][n]['portfolio']['annualized_excess_return_before_cost'])}</td>"
        f"<td>{pct(experiment['test_results'][n]['portfolio']['annualized_excess_return_after_cost'])}</td>"
        f"<td>{experiment['test_results'][n]['portfolio']['information_ratio_after_cost']:.3f}</td>"
        f"<td>{pct(experiment['test_results'][n]['portfolio']['max_drawdown_after_cost'])}</td></tr>"
        for n in METHODS
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qlib Factor Robustness Lab | Technical Case Study</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='18' fill='%23174a7e'/><text x='50' y='68' text-anchor='middle' font-size='56' fill='white'>Q</text></svg>">
<style>
:root{{--ink:#132238;--muted:#5b6b7f;--line:#d7dee8;--blue:#174a7e;--red:#b42318;--green:#067647;--paper:#f5f7fa}}*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,'Microsoft YaHei',sans-serif;line-height:1.65}}
nav{{position:sticky;top:0;z-index:5;background:rgba(19,34,56,.97);color:white}}nav .inner{{max-width:1180px;margin:auto;display:flex;gap:22px;align-items:center;padding:11px 24px;font-size:13px}}nav a{{color:#dbe7f3;text-decoration:none}}nav b{{margin-right:auto;letter-spacing:.04em}}
main{{max-width:1180px;margin:auto;padding:58px 24px 90px}}.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--blue);font-weight:800}}h1{{font-size:48px;line-height:1.1;margin:12px 0 18px;letter-spacing:-.03em}}h2{{font-size:28px;margin:58px 0 18px}}h3{{font-size:18px;margin:20px 0 10px}}.lead{{max-width:920px;font-size:19px;color:var(--muted)}}
.status{{margin:32px 0;padding:24px 28px;background:white;border:1px solid var(--line);border-left:7px solid var(--red);display:grid;grid-template-columns:1fr auto;gap:20px}}.status strong{{font-size:23px}}.decision{{font-size:30px;color:var(--red);font-weight:900}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,.panel{{background:white;border:1px solid var(--line);padding:20px}}.card{{min-height:126px}}.card .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}.card .v{{font-size:26px;font-weight:800;margin:5px 0}}.card p,.muted{{color:var(--muted)}}.card p{{font-size:13px;margin:0}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.panel.accent{{border-top:4px solid var(--blue)}}
figure{{margin:18px 0;background:white;border:1px solid var(--line);padding:16px}}figure img{{width:100%;display:block}}figcaption{{font-size:12px;color:var(--muted);margin-top:10px}}
table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th{{background:#e8eef5;text-align:left}}th,td{{border:1px solid var(--line);padding:11px 12px;vertical-align:top}}.table-wrap{{overflow-x:auto}}.pill{{font-size:11px;font-weight:800;padding:3px 8px;border-radius:99px}}.pass{{background:#d1fadf;color:var(--green)}}.fail{{background:#fee4e2;color:var(--red)}}
.formula{{font-family:'Cambria Math',serif;background:#f1f5f9;border-left:4px solid #0f7490;padding:16px 18px;margin:12px 0;font-size:16px}}.callout{{padding:18px 20px;background:#fff7ed;border:1px solid #fed7aa}}code{{background:#e8eef5;padding:2px 5px}}footer{{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}}
@media(max-width:850px){{h1{{font-size:36px}}.status{{grid-template-columns:1fr}}.decision{{font-size:24px}}.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}nav a{{display:none}}}}@media print{{nav{{display:none}}body{{background:white}}main{{padding:12mm}}figure,.panel,.card{{break-inside:avoid}}}}
</style></head><body><nav><div class="inner"><b>QFRL / TECHNICAL CASE</b><a href="#architecture">系统</a><a href="#protocol">协议</a><a href="#evidence">证据</a><a href="#gate">Gate</a><a href="interview-brief-v3.pdf">PDF</a></div></nav>
<main><div class="eyebrow">Quant Research Engineering · Case Study v3</div><h1>Qlib Factor Robustness Lab</h1>
<p class="lead">该项目是一套基于 Microsoft Qlib 的量化研究验证系统，用于检验因子信号能否从统计相关性跨越到含交易约束的组合结果。项目由两条证据链组成：上游基线复现用于验证研究环境，扩展实验用于验证时间安全、参数治理和研究晋级规则。</p>
<section class="status"><div><strong>结论：扩展方案未通过研究晋级 Gate</strong><div class="muted">信号层存在正 Rank IC，但成本后收益、信息比率、回撤和相对对照优势均不满足预设标准。结果被保留为失败诊断，而非包装为策略改进。</div></div><div class="decision">{gate['decision']} · {gate['passed_checks']}/{gate['total_checks']}</div></section>
<div class="grid"><div class="card"><div class="k">数据覆盖</div><div class="v">{experiment['data_manifest']['feature_rows']:,}</div><p>面板行；{experiment['data_manifest']['instruments']} 个历史成分证券</p></div><div class="card"><div class="k">冻结测试</div><div class="v">868 日</div><p>2017-2020；参数仅由 2016 验证期确定</p></div><div class="card"><div class="k">研究治理</div><div class="v">12 候选</div><p>验证期一次选参；两交易日信息滞后</p></div><div class="card"><div class="k">质量证据</div><div class="v">11 tests</div><p>核心覆盖率 86%；配置 SHA-256 留痕</p></div></div>
<h2 id="architecture">1. 系统边界与架构</h2><figure><img src="assets/v3_architecture.png" alt="四层研究系统架构"><figcaption>图 1｜数据、研究、执行和治理被拆成四个可独立追问的层级。Qlib 上游能力与项目扩展代码分开记录。</figcaption></figure>
<div class="two"><div class="panel accent"><h3>证据链 A：上游工作流复现</h3><p>Alpha158 + LightGBM + TopkDropoutStrategy 在锁定环境中完整运行。Rank IC 为 <b>{official['signal']['Rank IC']:.4f}</b>，成本后年化超额为 <b>{pct(official['portfolio']['annualized_excess_return_with_cost'])}</b>。</p><p class="muted">该结果只证明上游基线与本地环境可运行，不作为扩展方案的“改进前”收益。</p></div><div class="panel accent"><h3>证据链 B：稳定性加权扩展评估</h3><p>三类因子组合在相同测试区间、持仓规则和交易成本下接受对照。稳定性收缩方案成本后年化超额为 <b>{pct(experiment['test_results']['adaptive_shrinkage']['portfolio']['annualized_excess_return_after_cost'])}</b>，研究晋级状态为 <b>BLOCK</b>。</p><p class="muted">两条证据链目的不同，避免把不可直接比较的模型结果拼成收益提升叙事。</p></div></div>
<h2 id="protocol">2. 研究协议与时间可得性</h2><figure><img src="assets/v3_temporal_protocol.png" alt="训练验证测试与标签可得时间"><figcaption>图 2｜时间顺序切分和交易日 purge。T 日决策只使用截至 T-2 已实现的 IC。</figcaption></figure>
<div class="two"><div class="panel"><h3>信号构造</h3><div class="formula">IC<sub>j,t</sub> = Spearman(F<sub>j,t</sub>, R<sub>t+1→t+2</sub>)</div><div class="formula">w<sub>j,t</sub> ∝ sign(μ<sub>j,t</sub>) · max(|μ<sub>j,t</sub>| − λ·SE<sub>j,t</sub>, 0)</div><p class="muted">动态权重与训练期固定方向按验证期确定的比例收缩，并按绝对权重和归一化。</p></div><div class="panel"><h3>参数治理</h3><ul><li>训练期仅用于筛选 30 个因子及确定静态方向；</li><li>验证期比较 2 个窗口 × 2 个不确定性惩罚 × 3 个收缩比例；</li><li>测试期参数冻结，不依据测试结果二次选择；</li><li>三组信号进入同一 Qlib 执行器。</li></ul></div></div>
<figure><img src="assets/v3_validation_surface.png" alt="验证期候选参数面"><figcaption>图 3｜验证期候选配置完整展示，减少只报告最佳结果的选择性陈述风险。</figcaption></figure>
<h2 id="evidence">3. 样本外执行证据</h2><figure><img src="assets/v3_execution_diagnostics.png" alt="累计超额、成本、回撤与IC诊断"><figcaption>图 4｜信号质量、成本侵蚀、路径风险和可交易结果被同时报告。</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>扩展方案</th><th>Rank IC</th><th>ICIR</th><th>成本前超额</th><th>成本后超额</th><th>成本后 IR</th><th>最大回撤</th></tr></thead><tbody>{result_html}</tbody></table></div>
<div class="callout"><b>失败诊断：</b>三组扩展信号均具有正的截面排序相关性，但组合端持续落后于基准。执行成本造成约 4.5-4.8 个百分点的年化侵蚀；稳定性收缩方案同时出现更高换手、更低成本前收益和更深回撤。因此，失败不能只归因于手续费，更可能涉及尾部选股质量、组合集中和信号稳定性不足。</div>
<h2 id="gate">4. 机器可读研究晋级 Gate</h2><p>Gate 的范围是“是否进入更高保真研究阶段”，并非生产或实盘批准。阈值在 YAML 中固定，页面只读取运行结果。</p><div class="table-wrap"><table><thead><tr><th>检查项</th><th>观测值</th><th>阈值</th><th>结果</th></tr></thead><tbody>{check_html}</tbody></table></div>
<div class="two" style="margin-top:18px"><div class="panel"><h3>通过的证据</h3><p>测试样本长度、Rank IC 与 Rank ICIR 满足最低研究条件，表明信号并非完全随机。</p></div><div class="panel"><h3>阻断原因</h3><p>成本后超额、信息比率、最大回撤以及相对两个对照的收益与 IR 优势均未达标。因此系统不允许把该扩展称为“策略改进”。</p></div></div>
<h2>5. 可复现性、边界与下一阶段</h2><div class="two"><div class="panel"><h3>可复现证据</h3><ul><li>Python {experiment['runtime']['python']} / pyqlib {experiment['runtime']['pyqlib']} / LightGBM {experiment['runtime']['lightgbm']}</li><li>配置哈希：<code>{experiment['runtime']['config_sha256'][:16]}…</code></li><li>11 项测试；核心模块覆盖率 86%</li><li>CSV 路径、JSON 汇总、HTML 与 PDF 均由脚本生成</li></ul></div><div class="panel"><h3>下一阶段研究假设</h3><ol><li>把行业和规模暴露约束加入组合层；</li><li>显式约束换手并检验不同持有期；</li><li>分解 Top50 尾部选择与全截面 IC 的偏差；</li><li>在更可靠、更新的数据上重新校准结论。</li></ol></div></div>
<footer>数据为社区维护的 Qlib 示例数据，下载器说明底层价格来自 Yahoo Finance。材料仅用于教育、工程验证与求职展示，不构成投资建议或未来收益声明。上游许可和复用边界见 UPSTREAM.md。</footer></main></body></html>"""
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    v3_html = html.replace('src="assets/', 'src="../assets/')
    (DOCS / "v3" / "index.html").write_text(v3_html, encoding="utf-8")


def pdf_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=styles["Title"], fontName="MSYH-Bold",
                                fontSize=21, leading=27, alignment=TA_LEFT,
                                textColor=colors.HexColor(INK)),
        "subtitle": ParagraphStyle("Subtitle", parent=styles["Heading2"], fontName="MSYH-Bold",
                                   fontSize=12.5, leading=18, textColor=colors.HexColor(BLUE),
                                   spaceAfter=5),
        "h2": ParagraphStyle("H2CN", parent=styles["Heading2"], fontName="MSYH-Bold",
                             fontSize=13, leading=18, spaceBefore=5, spaceAfter=4,
                             textColor=colors.HexColor(BLUE)),
        "body": ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="MSYH",
                               fontSize=8.7, leading=13.2, textColor=colors.HexColor("#34465c")),
        "small": ParagraphStyle("SmallCN", parent=styles["BodyText"], fontName="MSYH",
                                fontSize=7.2, leading=10.4, textColor=colors.HexColor(MUTED)),
        "center": ParagraphStyle("CenterCN", parent=styles["BodyText"], fontName="MSYH-Bold",
                                 fontSize=9, leading=13, alignment=TA_CENTER,
                                 textColor=colors.HexColor(INK)),
    }


def styled_table(data: list[list[object]], widths: list[float], font_size: float = 7.5) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="MSYH-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.HexColor(INK),
    )
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="MSYH",
        fontSize=font_size,
        leading=font_size + 2.2,
        textColor=colors.HexColor(INK),
    )
    pass_style = ParagraphStyle("TablePass", parent=cell_style, fontName="MSYH-Bold",
                                textColor=colors.HexColor(GREEN))
    fail_style = ParagraphStyle("TableFail", parent=cell_style, fontName="MSYH-Bold",
                                textColor=colors.HexColor(RED))
    limited_style = ParagraphStyle("TableLimited", parent=cell_style, fontName="MSYH-Bold",
                                   textColor=colors.HexColor(GOLD))

    def style_for(value: object, row_index: int) -> ParagraphStyle:
        if row_index == 0:
            return header_style
        if str(value) == "PASS":
            return pass_style
        if str(value) in {"FAIL", "BLOCK"}:
            return fail_style
        if str(value) == "LIMITED":
            return limited_style
        return cell_style

    wrapped = [
        [Paragraph(html.escape(str(value)), style_for(value, row_index)) for value in row]
        for row_index, row in enumerate(data)
    ]
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "MSYH", font_size),
        ("FONT", (0, 0), (-1, 0), "MSYH-Bold", font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4ebf3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf(experiment: dict, official: dict) -> Path:
    pdfmetrics.registerFont(TTFont("MSYH", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", str(FONT_BOLD)))
    output = PDF_DIR / "interview-brief-v3.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=12 * mm)
    s = pdf_styles()
    gate = experiment["research_promotion_gate"]
    story: list[object] = []

    story += [
        Paragraph("Qlib Factor Robustness Lab", s["title"]),
        Paragraph("量化研究验证系统｜Technical Case Study v3", s["subtitle"]),
        Paragraph("该项目检验因子信号能否从统计相关性跨越到含交易约束的组合结果。证据链分为上游基线复现与扩展方案评估，避免把不可直接比较的结果包装成收益提升。", s["body"]),
        Spacer(1, 3 * mm),
    ]
    executive = [["最终状态", "Gate 证据", "测试期", "工程验证"],
                 [gate["decision"], f"{gate['passed_checks']} / {gate['total_checks']} 通过",
                  "2017-2020 / 868 标签日", "11 tests / 86% coverage"]]
    summary_table = styled_table(executive, [40 * mm, 44 * mm, 51 * mm, 45 * mm], 8)
    summary_table.setStyle(TableStyle([("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor(RED)),
                                       ("FONT", (0, 1), (0, 1), "MSYH-Bold", 11)]))
    evidence = [
        ["证据链", "目的", "关键结果", "可得结论"],
        ["A / 上游复现", "验证 Alpha158 + LightGBM + Qlib 执行链路",
         f"Rank IC {official['signal']['Rank IC']:.4f}; 成本后年化超额 {pct(official['portfolio']['annualized_excess_return_with_cost'])}",
         "基线与环境可运行"],
        ["B / 扩展评估", "验证稳定性加权、时间治理与研究 Gate",
         f"Rank IC {experiment['test_results']['adaptive_shrinkage']['signal']['rank_ic_mean']:.4f}; 成本后年化超额 {pct(experiment['test_results']['adaptive_shrinkage']['portfolio']['annualized_excess_return_after_cost'])}",
         "未达到研究晋级标准"],
    ]
    story += [summary_table, Spacer(1, 4 * mm), Paragraph("两条证据链", s["h2"]),
              styled_table(evidence, [31 * mm, 49 * mm, 60 * mm, 40 * mm], 7.4),
              Spacer(1, 4 * mm), Paragraph("决策摘要", s["h2"]),
              Paragraph("扩展方案在信号层具有正 Rank IC，但成本后收益、信息比率、回撤及相对对照优势均未达标。系统输出 BLOCK，并保留失败路径、候选参数面和机器检查结果。项目价值位于研究工程与模型风险治理，而非收益承诺。", s["body"]),
              Spacer(1, 3 * mm), Image(str(ASSETS / "v3_architecture.png"), width=180 * mm, height=60 * mm),
              PageBreak()]

    controls = [
        ["研究风险", "控制设计", "可核验证据", "状态"],
        ["标签信息提前使用", "IC 权重滞后 2 个交易日", "temporal.py + 边界测试", "PASS"],
        ["训练/验证边界污染", "交易日 purge；有效截止日留痕", "results.json / timeline", "PASS"],
        ["测试集反复调参", "12 个候选只在 2016 验证期比较", "候选面完整输出", "PASS"],
        ["成本与执行简化", "统一 Topk/Drop5/涨跌停/双边成本", "3 组 backtest CSV", "PASS"],
        ["选择性报告", "三组方案、失败路径、回撤与成本同时披露", "HTML/PDF/JSON", "PASS"],
        ["研究结论越级", "8 项机器 Gate；限定为研究晋级", "gating.py + 2 tests", "PASS"],
        ["数据外推风险", "社区示例数据仅作工程验证", "UPSTREAM.md / disclaimer", "LIMITED"],
    ]
    story += [Paragraph("01｜系统架构与责任边界", s["title"]),
              Paragraph("四层架构将数据、研究、执行与治理证据拆分，支持逐层审阅。", s["body"]),
              Image(str(ASSETS / "v3_architecture.png"), width=180 * mm, height=60 * mm),
              Paragraph("关键风险—控制—证据矩阵", s["h2"]),
              styled_table(controls, [38 * mm, 61 * mm, 54 * mm, 27 * mm], 7.2),
              Spacer(1, 4 * mm), Paragraph("上游与扩展边界", s["h2"]),
              Paragraph("Microsoft Qlib 提供数据格式、Alpha158、LightGBM 工作流、TopkDropoutStrategy 与交易模拟器。项目代码负责截面 IC、信息可得滞后、交易日边界、稳定性权重、参数治理、研究 Gate、自动化证据和展示生成。上游源代码未被修改。", s["body"]),
              Paragraph("数据规模与运行环境", s["h2"]),
              Paragraph(f"{experiment['data_manifest']['feature_rows']:,} 行；{experiment['data_manifest']['instruments']} 个历史成分证券；Python {experiment['runtime']['python']}；pyqlib {experiment['runtime']['pyqlib']}；LightGBM {experiment['runtime']['lightgbm']}；配置哈希 {experiment['runtime']['config_sha256'][:20]}…", s["body"]),
              PageBreak()]

    story += [Paragraph("02｜研究协议与参数治理", s["title"]),
              Paragraph("时间顺序切分、标签可得性和一次性参数选择构成主要识别边界。", s["body"]),
              Image(str(ASSETS / "v3_temporal_protocol.png"), width=180 * mm, height=56 * mm),
              Paragraph("方法定义", s["h2"]),
              Paragraph("每日截面 Rank IC：IC[j,t] = Spearman(F[j,t], R[t+1→t+2])。动态权重使用滞后 IC 的滚动均值减去估计标准误惩罚：w[j,t] ∝ sign(μ[j,t]) × max(|μ[j,t]| - λ × SE[j,t], 0)，随后与训练期固定方向收缩并归一化。", s["body"]),
              Paragraph("验证期候选面", s["h2"]),
              Image(str(ASSETS / "v3_validation_surface.png"), width=166 * mm, height=67 * mm),
              Paragraph("候选空间包含窗口 60/120、惩罚 0.5/1.0、动态权重占比 25%/50%/75%，共 12 项。验证期选择 window=60、penalty=1.0、dynamic share=75%；测试期不再调参。", s["body"]),
              PageBreak()]

    story += [Paragraph("03｜样本外执行证据", s["title"]),
              Paragraph("三组扩展方案使用同一股票池、持仓与交易规则，避免执行假设不一致。", s["body"]),
              Image(str(ASSETS / "v3_execution_diagnostics.png"), width=180 * mm, height=110 * mm),
              styled_table(result_rows(experiment),
                           [32 * mm, 19 * mm, 18 * mm, 24 * mm, 24 * mm, 18 * mm, 24 * mm, 21 * mm], 6.5),
              Spacer(1, 3 * mm), Paragraph("诊断结论", s["h2"]),
              Paragraph("正 Rank IC 未转化为成本后组合收益。执行成本约造成 4.5-4.8 个百分点的年化侵蚀，但三组方案在成本前已为负，因此手续费不是唯一原因。稳定性收缩方案相较静态对照具有更低的成本前收益、更高换手和更深回撤，说明动态加权在该样本中增加了估计噪声与组合不稳定性。", s["body"]),
              PageBreak()]

    roadmap = [
        ["优先级", "假设", "实验设计", "继续条件"],
        ["P1", "尾部 Top50 与全截面 IC 不一致", "分位数组合与 top-k 敏感性分析", "尾部单调性稳定"],
        ["P1", "行业/规模暴露主导组合", "加入行业中性与规模约束", "风险调整后超额改善"],
        ["P2", "动态权重增加换手和噪声", "显式换手惩罚与持有期网格", "成本后 IR 转正"],
        ["P2", "示例数据限制结论外推", "更可靠数据上的滚动重估", "跨时期方向一致"],
    ]
    story += [Paragraph("04｜研究晋级决策与后续路线", s["title"]),
              Paragraph("Gate 只决定是否进入更高保真研究阶段，不代表生产或实盘批准。", s["body"]),
              styled_table(gate_rows(experiment), [64 * mm, 39 * mm, 39 * mm, 38 * mm], 7.5),
              Spacer(1, 4 * mm),
              Paragraph(f"最终决策：{gate['decision']}（{gate['passed_checks']} / {gate['total_checks']} 项通过）", s["h2"]),
              Paragraph("通过项说明信号具有最低统计研究价值；失败项说明证据不足以支持进一步部署。成本后超额、信息比率、最大回撤及相对两个对照的收益和 IR 优势均未达标，因此扩展不能被称为策略改进。", s["body"]),
              Paragraph("下一阶段研究路线", s["h2"]),
              styled_table(roadmap, [23 * mm, 53 * mm, 66 * mm, 38 * mm], 7.1),
              Spacer(1, 4 * mm), Paragraph("审阅入口", s["h2"]),
              Paragraph("README：项目全景｜docs/index.html：浏览器技术案例｜evidence/：机器结果与清单｜scripts/run_portfolio_v2.py：端到端实验｜src/qlib_factor_lab/：独立研究模块｜tests/：时间边界、权重与 Gate 测试。", s["body"]),
              Paragraph("边界声明", s["h2"]),
              Paragraph("社区 Qlib 示例数据的底层来源被下载器标注为 Yahoo Finance；该数据不等同于机构级行情。材料只支持教育、研究工程与求职展示，不构成投资建议、实盘批准或未来收益声明。", s["small"])]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setTitle("Qlib Factor Robustness Lab - Technical Case Study v3")
        canvas.setAuthor("Tian Yihan")
        canvas.setSubject("Quant research engineering and model-risk case study")
        canvas.setFont("MSYH", 7.2)
        canvas.setFillColor(colors.HexColor("#7b8da3"))
        canvas.drawString(15 * mm, 7 * mm, "Qlib Factor Robustness Lab | Technical Case Study v3")
        canvas.drawRightString(A4[0] - 15 * mm, 7 * mm, f"{document.page} / 5")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def publish_evidence(experiment: dict) -> None:
    keys = ["status", "runtime", "data_manifest", "temporal_controls",
            "selected_adaptive_parameters", "test_results", "research_promotion_gate",
            "interpretation_rule"]
    (EVIDENCE / "v2_verified_results.json").write_text(
        json.dumps({key: experiment[key] for key in keys}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "showcase_version": "v3",
        "methodology_result": "artifacts/portfolio_v2/results.json",
        "methodology_sha256": hashlib.sha256((ARTIFACTS / "results.json").read_bytes()).hexdigest(),
        "gate_decision": experiment["research_promotion_gate"]["decision"],
        "generated_outputs": [
            "docs/index.html", "docs/v3/index.html", "docs/assets/v3_architecture.png",
            "docs/assets/v3_temporal_protocol.png", "docs/assets/v3_validation_surface.png",
            "docs/assets/v3_execution_diagnostics.png", "output/pdf/interview-brief-v3.pdf",
        ],
    }
    (EVIDENCE / "v3_showcase_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    setup_dirs()
    experiment, official = load()
    font = FontProperties(fname=str(FONT_REGULAR))
    plot_architecture(experiment, font)
    plot_temporal_protocol(experiment, font)
    plot_validation_surface(experiment, font)
    plot_execution_diagnostics(experiment, font)
    build_html(experiment, official)
    publish_evidence(experiment)
    pdf = build_pdf(experiment, official)
    shutil.copyfile(pdf, DOCS / "interview-brief-v3.pdf")
    shutil.copyfile(pdf, DOCS / "v3" / "interview-brief-v3.pdf")
    print(pdf)


if __name__ == "__main__":
    main()
