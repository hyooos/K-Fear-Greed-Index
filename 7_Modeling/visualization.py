import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import TARGET_DAILY_VOL, CFG


def setup_matplotlib():
    import sys
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def calc_dd(ret):
    c = np.exp(ret.cumsum())
    return (c / c.cummax() - 1) * 100


def plot_main_results(df, main_df, trend_df, trendvol_df, df_ns, main_ns,
                      rho_5d, kdir, save_path="kfgi_egarch_result_v5.png"):
    dates      = df["date"]
    shock_mask = df["vol_shock"] > CFG["vol_shock_thr"]
    ref        = df["target_reg"]

    fig = plt.figure(figsize=(17, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Panel 1 — 누적 수익률
    ax1.plot(dates, np.exp(ref.cumsum()),
             label="KOSPI B&H", color="#9CA3AF", lw=1.5, alpha=0.8)
    ax1.plot(dates, np.exp(trend_df["strat_ret"].cumsum()),
             label="① 추세추종", color="#16A34A", lw=1.5, ls="-.")
    ax1.plot(dates, np.exp(trendvol_df["strat_ret"].cumsum()),
             label="② 추세+vol^0.5", color="#F59E0B", lw=1.5, ls="--")
    ax1.plot(dates, np.exp(main_df["strat_ret"].cumsum()),
             label=f"③ 메인 v5 (K-FGI {kdir})", color="#2563EB", lw=2.5)
    ax1.plot(df_ns["date"], np.exp(main_ns["strat_ret"].cumsum()),
             label="③ 메인 v5 (감성 제외)", color="#7C3AED",
             lw=1.2, ls=":", alpha=0.7)
    ax1.fill_between(dates, 0.5, 4, where=shock_mask,
                     color="#FCA5A5", alpha=0.2, label="vol_shock 경보")
    ax1.set_title(
        f"누적 수익률 v5  (K-FGI {kdir}, ρ(5d)={rho_5d:.3f}, 배율 0.5~1.6)",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(alpha=0.2)

    # Panel 2 — Drawdown
    for ret, lbl, col, alp in [
        (ref,                      "KOSPI",        "#9CA3AF", 0.40),
        (trend_df["strat_ret"],    "추세추종",      "#16A34A", 0.35),
        (trendvol_df["strat_ret"], "추세+vol^0.5", "#F59E0B", 0.35),
        (main_df["strat_ret"],     "메인 v5",       "#2563EB", 0.50),
    ]:
        ax2.fill_between(dates, calc_dd(ret), color=col, alpha=alp, label=lbl)
    ax2.set_title("Drawdown 비교 (%)", fontsize=11)
    ax2.set_ylabel("DD (%)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # Panel 3 — EGARCH σ vs vol-targeting pos
    raw_vt_plot  = TARGET_DAILY_VOL / (df["egarch_vol"] + 1e-9)
    vt_soft_plot = np.sqrt(raw_vt_plot).clip(CFG["min_leverage"], CFG["max_leverage"])
    ax3.fill_between(dates, df["egarch_vol"] * 100,
                     alpha=0.5, color="#F59E0B", label="EGARCH σ (%)")
    ax3_r = ax3.twinx()
    ax3_r.plot(dates, vt_soft_plot, color="#2563EB", lw=1.2,
               label="vol-target pos (^0.5)")
    ax3_r.axhline(1.0, color="#6B7280", linestyle=":", lw=0.8)
    ax3_r.set_ylabel("포지션 배수")
    ax3.set_title("EGARCH σ vs vol-targeting pos (^0.5)", fontsize=11)
    ax3.set_ylabel("σ (%)")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax3.grid(alpha=0.2)

    # Panel 4 — K-FGI
    kv = df["K_FGI"].values
    ax4.plot(dates, kv, color="#7C3AED", lw=1.0)
    ax4.axhline(25, color="#2563EB", linestyle="--", lw=0.8, alpha=0.7, label="K-FGI<25 (공포)")
    ax4.axhline(75, color="#DC2626", linestyle="--", lw=0.8, alpha=0.7, label="K-FGI>75 (탐욕)")
    ax4.fill_between(dates, 0, 100, where=(kv < 25),
                     color="#BFDBFE", alpha=0.45, label="공포 구간")
    ax4.fill_between(dates, 0, 100, where=(kv > 75),
                     color="#FCA5A5", alpha=0.35, label="탐욕 구간")
    ax4.set_title(f"K-FGI (순수 감성+변동성, Ridge 무제약, {kdir}, 배율 0.5~1.6)", fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=7.5)
    ax4.grid(alpha=0.2)

    # Panel 5 — 포지션 크기
    ax5.plot(dates, main_df["weight"],    color="#2563EB", lw=0.9, label="메인 v5")
    ax5.plot(dates, trend_df["weight"],   color="#16A34A", lw=0.9,
             alpha=0.7, label="추세추종", ls="-.")
    ax5.plot(df_ns["date"], main_ns["weight"], color="#7C3AED", lw=0.9,
             alpha=0.7, label="감성 제외", ls=":")
    ax5.axhline(1.0, color="#6B7280", linestyle=":", lw=0.8)
    ax5.set_title("포지션 크기 (vol^0.5 + K-FGI 0.5~1.6 + crisis 0.7)", fontsize=11)
    ax5.set_ylabel("레버리지 배수")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.2)

    fig.suptitle(
        f"K-FGI × EGARCH v5 — vol^0.5 | crisis 0.7 | K-FGI 0.5~1.6 | {kdir}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n시각화 저장: {save_path}")