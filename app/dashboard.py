"""
Alpha158 A 股选股研究面板(Streamlit)

启动：
    source /home/oracle/vnpy-venv/bin/activate
    streamlit run app/dashboard.py

说明：界面隔离在 app/ 目录，scripts/ 保持纯研究脚本(遵守 CLAUDE.md 约束)。
展示选股清单/方案对比/收益曲线，并提供轻量触发(出最新信号)。
训练/回测耗时长，界面给出命令行命令,不在 web 同步执行。
"""

import json
import subprocess
from pathlib import Path

import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab_data"
APP = Path(__file__).resolve().parent
VENV_PY = "/home/oracle/vnpy-venv/bin/python"

st.set_page_config(page_title="Alpha158 选股研究", layout="wide")
st.title("📈 Alpha158 A 股选股研究面板")
st.caption("中盘 CSI500P + Alpha158 + point-in-time 财务因子 | 最优 Sharpe 0.85、超额 +44.7%")

tab_picks, tab_cmp, tab_curve, tab_run = st.tabs(
    ["📋 选股清单", "📊 方案对比", "📈 收益曲线", "▶️ 运行"]
)

# ---------------- 选股清单 ----------------
with tab_picks:
    csvs = sorted(LAB.glob("picks_*.csv"))
    if not csvs:
        st.info("暂无清单。到「运行」Tab 点击生成,或命令行跑 generate_signal.py")
    else:
        latest = csvs[-1]
        st.subheader(f"最新清单: {latest.stem.replace('picks_', '')}")
        df = pl.read_csv(latest).to_pandas()
        st.dataframe(df, use_container_width=True, height=600)
        st.download_button("下载 CSV", latest.read_bytes(), file_name=latest.name)
        st.caption("⚠️ 研究输出,非投资建议;T 日收盘信号 → T+1 开盘买入")

# ---------------- 方案对比 ----------------
with tab_cmp:
    summary = json.loads((APP / "results_summary.json").read_text())
    st.caption(f"{summary['oos_period']} | {summary['benchmark']} | 更新 {summary['updated']}")

    schemes = summary["schemes"]
    best = max(schemes, key=lambda s: s["sharpe"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最优方案", best["name"])
    c2.metric("Sharpe", best["sharpe"])
    c3.metric("超额收益", f"{best['excess']:+.1%}")
    c4.metric("收益回撤比", best["calmar"])

    rows = []
    for s in schemes:
        rows.append({
            "方案": s["name"], "因子": s["factors"],
            "Sharpe": s["sharpe"], "超额": f"{s['excess']:+.1%}",
            "年化": f"{s['annual']:+.1%}", "最大回撤": f"{s['maxdd']:.1%}",
            "收益回撤比": s["calmar"], "结论": s["note"],
        })
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)
    st.info("核心洞察:攻因子要选对维度 —— 估值(pe/pb)已被量价隐含无用;成长/质量(营收增速/ROE)正交有效,突破 0.78 天花板。")

# ---------------- 收益曲线 ----------------
with tab_curve:
    results_dir = LAB / "results"
    curves = sorted(results_dir.glob("*_curve.parquet")) if results_dir.exists() else []
    if not curves:
        st.info("暂无曲线。跑回测会自动生成 (alpha158_backtest.py)。")
    else:
        names = [c.stem.replace("_curve", "") for c in curves]
        sel = st.selectbox("选择方案", names)
        cdf = pl.read_parquet(results_dir / f"{sel}_curve.parquet").to_pandas().set_index("date")
        st.line_chart(cdf[["strat", "bench"]].rename(columns={"strat": "策略净值", "bench": "沪深300"}))
        final = cdf["strat"].iloc[-1] - 1
        st.metric("期末累计收益(策略)", f"{final:+.1%}")

# ---------------- 运行 ----------------
with tab_run:
    st.subheader("生成最新选股清单(秒级)")
    top_k = st.slider("选股数量 top_k", 10, 100, 50, 10)
    if st.button("🚀 生成清单", type="primary"):
        with st.spinner("生成中..."):
            r = subprocess.run(
                [VENV_PY, "scripts/generate_signal.py", "--top-k", str(top_k)],
                capture_output=True, text=True, cwd=str(ROOT), timeout=180,
            )
        if r.returncode == 0:
            st.success("完成!到「选股清单」Tab 查看")
            st.code(r.stdout[-3000:])
        else:
            st.error(r.stderr[-2000:])

    st.divider()
    st.subheader("训练 / 回测(耗时长,请命令行运行)")
    st.code(
        "# 训练(中盘+财务因子, ~28分钟)\n"
        "python scripts/alpha158_train.py --name a158_csi500p_fin \\\n"
        "    --universe CSI500P --batch-size 200 --workers 6 --financial\n\n"
        "# 滚动验证\n"
        "python scripts/alpha158_rolling.py --base a158_csi500p_fin\n\n"
        "# 回测(自动生成收益曲线)\n"
        "python scripts/alpha158_backtest.py --name a158_csi500p_fin_rolling \\\n"
        "    --start 2023-01-01 --end 2026-05-29 --t1 --filter-limit",
        language="bash",
    )
