"""Informe local de datos y modelo: HTML autonomo, Markdown, PDF, PNG/SVG y CSV."""

import argparse
import base64
import csv
import html
import os
import textwrap
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aivr-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from app import features
from app.interventions import LAYER_NAMES, TURN_GROUPS, feature_names

COLORS = {"human": "#087e8b", "synthetic": "#da6c37"}
GROUP_COLORS = dict(
    zip(
        LAYER_NAMES, ["#d99a24", "#8064a2", "#428bb4", "#159383", "#cf607c", "#60748c"], strict=True
    )
)
BUDGET_NAMES = {
    "first_turn": "Primera intervención",
    "20s": "Primeros 20 s",
    "full": "Llamada completa",
}
STRESS_NAMES = ("humano_limpio", "bot_evasivo", "bot_ritmo")


def table_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe_matrix(X, names, unit):
    rows = []
    for j, name in enumerate(names):
        finite = X[np.isfinite(X[:, j]), j]
        quartiles = (
            np.quantile(finite, [0, 0.005, 0.25, 0.5, 0.75, 0.995, 1])
            if len(finite)
            else [np.nan] * 7
        )
        iqr = quartiles[4] - quartiles[2]
        low, high = quartiles[2] - 1.5 * iqr, quartiles[4] + 1.5 * iqr
        extreme = int(((finite < low) | (finite > high)).sum()) if iqr > 0 else 0
        rows.append(
            {
                "unit": unit,
                "feature": name,
                "group": features.grupo_de(name),
                "n": len(X),
                "missing": int(np.isnan(X[:, j]).sum()),
                "infinite": int(np.isinf(X[:, j]).sum()),
                "missing_pct": float(100 * (~np.isfinite(X[:, j])).mean()),
                "unique": len(np.unique(finite)),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite)) if len(finite) else np.nan,
                **dict(
                    zip(
                        ("min", "p005", "q1", "median", "q3", "p995", "max"),
                        map(float, quartiles),
                        strict=True,
                    )
                ),
                "iqr_outliers": extreme,
                "iqr_outlier_pct": 100 * extreme / max(1, len(finite)),
                "zero_iqr": bool(iqr == 0),
            }
        )
    return rows


def correlation_analysis(X, y, names, output):
    n = len(names)
    pearson = np.full((n, n), np.nan)
    spearman = np.full((n, n), np.nan)
    associations, pairs = [], []
    for j, name in enumerate(names):
        valid = np.isfinite(X[:, j])
        if valid.sum() < 4 or len(np.unique(X[valid, j])) < 2:
            continue
        rho, p = stats.spearmanr(X[valid, j], y[valid])
        associations.append(
            {
                "feature": name,
                "group": features.grupo_de(name),
                "n": int(valid.sum()),
                "spearman_label": float(rho),
                "p_value": float(p),
                "median_human": float(np.nanmedian(X[y == 0, j])),
                "median_synthetic": float(np.nanmedian(X[y == 1, j])),
            }
        )
        for k in range(j, n):
            pair = valid & np.isfinite(X[:, k])
            if pair.sum() < 4 or len(np.unique(X[pair, k])) < 2 or len(np.unique(X[pair, j])) < 2:
                continue
            pr = float(stats.pearsonr(X[pair, j], X[pair, k]).statistic)
            sr = float(stats.spearmanr(X[pair, j], X[pair, k]).statistic)
            pearson[j, k] = pearson[k, j] = pr
            spearman[j, k] = spearman[k, j] = sr
            if k > j and abs(sr) >= 0.75:
                within = []
                for label in (0, 1):
                    m = pair & (y == label)
                    if m.sum() >= 4 and all(len(np.unique(X[m, col])) > 1 for col in (j, k)):
                        within.append(float(stats.spearmanr(X[m, j], X[m, k]).statistic))
                    else:
                        within.append(np.nan)
                pairs.append(
                    {
                        "feature_a": name,
                        "feature_b": names[k],
                        "spearman": sr,
                        "pearson": pr,
                        "n": int(pair.sum()),
                        "rho_human": within[0],
                        "rho_synthetic": within[1],
                    }
                )
    adjusted = stats.false_discovery_control([r["p_value"] for r in associations])
    for row, q in zip(associations, adjusted, strict=True):
        row["q_bh"] = float(q)
    associations.sort(key=lambda r: abs(r["spearman_label"]), reverse=True)
    pairs.sort(key=lambda r: abs(r["spearman"]), reverse=True)
    table_csv(output / "feature_label_correlations.csv", associations)
    table_csv(output / "strong_correlation_pairs.csv", pairs)
    for label, matrix in (("pearson", pearson), ("spearman", spearman)):
        with (output / f"correlation_{label}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", *names])
            for name, row in zip(names, matrix, strict=True):
                writer.writerow([name, *row])
    return associations, pairs, pearson, spearman


class Report:
    def __init__(self, output):
        self.output = output
        self.figures = output / "figures"
        self.figures.mkdir(exist_ok=True)
        self.blocks = []
        self.pdf_lines = []
        self.pdf = PdfPages(
            output / "reporte.pdf", metadata={"Title": "AIVoiceRecognition — Issue 5"}
        )

    def heading(self, text):
        self.blocks.append(("heading", text))
        self.pdf_lines.extend(["", text.upper(), ""])

    def paragraph(self, text):
        self.blocks.append(("paragraph", text))
        self.pdf_lines.extend(textwrap.wrap(text, 91) + [""])

    def table(self, headers, rows):
        self.blocks.append(("table", (headers, rows)))
        self.pdf_lines.extend(textwrap.wrap(" | ".join(headers), 91) + [""])
        for row in rows:
            self.pdf_lines.extend(textwrap.wrap(" | ".join(map(str, row)), 91))
        self.pdf_lines.append("")

    def flush_pdf_text(self):
        while self.pdf_lines:
            page, self.pdf_lines = self.pdf_lines[:44], self.pdf_lines[44:]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(
                0.08,
                0.95,
                "AIVoiceRecognition · análisis y evaluación",
                fontsize=12,
                weight="bold",
                va="top",
                color="#124f60",
            )
            fig.text(0.08, 0.90, "\n".join(page), fontsize=9.5, va="top", linespacing=1.5)
            self.pdf.savefig(fig)
            plt.close(fig)

    def figure(self, name, fig, caption):
        self.flush_pdf_text()
        fig.savefig(self.figures / f"{name}.png", dpi=155, bbox_inches="tight", facecolor="white")
        fig.savefig(self.figures / f"{name}.svg", bbox_inches="tight", facecolor="white")
        self.pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.blocks.append(("figure", (name, caption)))

    def text_page(self, title, paragraphs):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")
        body = "\n\n".join(textwrap.fill(p, 91) for p in paragraphs)
        fig.text(0.08, 0.88, body, fontsize=10, va="top", linespacing=1.6)
        self.pdf.savefig(fig)
        plt.close(fig)

    def finish(self):
        self.flush_pdf_text()
        self.pdf.close()
        parts, md = [], ["# Análisis del dataset y modelo del issue #5\n"]
        for kind, payload in self.blocks:
            if kind == "heading":
                parts.append(f"<h2>{html.escape(payload)}</h2>")
                md.append(f"\n## {payload}\n")
            elif kind == "paragraph":
                parts.append(f"<p>{html.escape(payload)}</p>")
                md.append(payload + "\n")
            elif kind == "table":
                headers, rows = payload
                parts.append(
                    "<div class='scroll'><table><thead><tr>"
                    + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
                    + "</tr></thead><tbody>"
                    + "".join(
                        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                        for row in rows
                    )
                    + "</tbody></table></div>"
                )
                md.append("| " + " | ".join(headers) + " |\n|" + " --- |" * len(headers))
                md.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
                md.append("")
            else:
                name, caption = payload
                encoded = base64.b64encode((self.figures / f"{name}.png").read_bytes()).decode()
                parts.append(
                    f"<figure><img src='data:image/png;base64,{encoded}' "
                    f"alt='{html.escape(caption, quote=True)}'><figcaption>"
                    f"{html.escape(caption)}</figcaption></figure>"
                )
                md.append(f"![{caption}](figures/{name}.png)\n")
        sources = [
            ("Dataset oficial", "https://github.com/alturio/hackmty26"),
            ("Issue #5", "https://github.com/samueldev-boop/AIVoiceRecognition/issues/5"),
            (
                "StratifiedGroupKFold",
                "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html",
            ),
            ("Calibración", "https://scikit-learn.org/stable/modules/calibration.html"),
            ("Preprocesamiento y fuga", "https://scikit-learn.org/stable/common_pitfalls.html"),
        ]
        parts.append(
            "<h2>Referencias y reproducción</h2><p>"
            + " · ".join(f"<a href='{url}'>{title}</a>" for title, url in sources)
            + "</p>"
        )
        command = "python -m scripts.train_issue5\npython -m scripts.report_issue5"
        parts.append(f"<pre>{command}</pre>")
        md.append("\n## Referencias y reproducción\n")
        md.extend(f"- [{title}]({url})" for title, url in sources)
        md.append(f"\n```bash\n{command}\n```\n")
        self.output.joinpath("reporte.md").write_text("\n".join(md))
        template = """<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Análisis del dataset · AIVoiceRecognition</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf2f4;color:#213445;
font:16px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:32px auto;background:white;
padding:48px 64px;border-radius:16px;box-shadow:0 8px 35px #17394c12}header{border-bottom:3px solid
#087e8b;padding-bottom:24px;margin-bottom:36px}h1{font-size:38px;line-height:1.15;max-width:850px}
.eyebrow{color:#087e8b;font-weight:700;letter-spacing:.1em;font-size:12px}h2{margin-top:48px;
font-size:25px;color:#124f60}p{max-width:98ch}a{color:#087e8b}table{border-collapse:collapse;
width:100%;font-size:13px;font-variant-numeric:tabular-nums}th{background:#eaf2f4;text-align:left}
td,th{padding:10px 12px;border-bottom:1px solid #e2e9ed}.scroll{overflow-x:auto;margin:24px 0}
figure{margin:28px 0 40px}img{width:100%;height:auto}figcaption{font-size:13px;color:#526677}
pre{padding:20px;background:#edf3f6;overflow:auto;border-radius:8px}
@media(max-width:750px){main{padding:24px;margin:0;border-radius:0}h1{font-size:29px}}
@media print{body{background:white}main{padding:0;box-shadow:none}figure,table{break-inside:avoid}}
</style><main><header><div class="eyebrow">ALTUR · HACKMTY 2026 · ISSUE #5</div>
<h1>De las intervenciones de voz a una decisión por llamada</h1>
<p>Auditoría, limpieza, correlaciones, validación agrupada y pruebas de estrés.</p></header>
"""
        self.output.joinpath("reporte.html").write_text(
            template + "\n".join(parts) + "</main></html>"
        )


def metric_rows(report, split="cv"):
    rows = []
    for budget, result in report["budgets"].items():
        m = result["cv"]["clean"] if split == "cv" else result["val"]
        threshold = m["thresholds"]["0.5"]
        rows.append(
            [
                BUDGET_NAMES[budget],
                m["n"],
                f"{m['auc']:.4f}",
                f"{100 * threshold['accuracy']:.2f}%",
                f"{100 * threshold['precision']:.2f}%",
                f"{100 * threshold['recall']:.2f}%",
                f"{threshold['f1']:.4f}",
                f"{m['brier']:.4f}",
            ]
        )
    return rows


def basic_plots(doc, train, all_records, X, y, names, associations, pearson, spearman):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    for j, split in enumerate(("train", "val")):
        counts = [
            sum(r["meta"]["split"] == split and r["meta"]["label"] == label for r in all_records)
            for label in COLORS
        ]
        bars = axes[0].bar(
            np.arange(2) + (j - 0.5) * 0.36,
            counts,
            0.36,
            label=split,
            color=("#087e8b", "#91c6cb")[j],
        )
        axes[0].bar_label(bars)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Humana", "Sintética"],
        ylabel="Llamadas",
        title="Clases y splits",
    )
    axes[0].legend()
    for label, color in COLORS.items():
        members = [r for r in train if r["meta"]["label"] == label]
        axes[1].hist(
            [r["audit"]["duration_s"] for r in members],
            bins=22,
            density=True,
            alpha=0.55,
            color=color,
            label=label,
        )
        durations = [v for r in members for v in r["audit"]["vad_caller_durations"]]
        axes[2].hist(
            durations,
            bins=np.geomspace(0.19, max(durations) + 0.1, 35),
            density=True,
            alpha=0.55,
            color=color,
            label=label,
        )
    axes[1].set(title="Duración de llamada · train", xlabel="Segundos", ylabel="Densidad")
    axes[2].set(title="Duración de turno VAD · train", xlabel="Segundos (escala log)", xscale="log")
    axes[1].legend()
    doc.figure(
        "dataset_distributions",
        fig,
        "Las duraciones se leen del WAV; los histogramas por turno son descriptivos.",
    )

    selected = []
    for group in LAYER_NAMES:
        selected.extend([r["feature"] for r in associations if r["group"] == group][:4])
    index = [names.index(k) for k in selected]
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    matrix = spearman[np.ix_(index, index)]
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(
        xticks=range(len(index)),
        yticks=range(len(index)),
        xticklabels=selected,
        yticklabels=selected,
        title="Correlación de Spearman · train · hasta 4 variables por grupo",
    )
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="ρ de Spearman")
    doc.figure(
        "heatmap_spearman",
        fig,
        "Selección descriptiva por asociación con la etiqueta; "
        "no modifica las features del modelo.",
    )

    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    im = ax.imshow(pearson[np.ix_(index, index)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(
        xticks=range(len(index)),
        yticks=range(len(index)),
        xticklabels=selected,
        yticklabels=selected,
        title="Correlación de Pearson · mismas variables · train",
    )
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="r de Pearson")
    doc.figure(
        "heatmap_pearson",
        fig,
        "Pearson mide relaciones lineales; Spearman compara rangos y reduce la influencia "
        "de las colas extremas. Ambas matrices completas se exportan a CSV.",
    )

    graph = nx.Graph()
    graph.add_nodes_from(selected)
    for i, a in enumerate(selected):
        for j, b in enumerate(selected[i + 1 :], i + 1):
            rho = matrix[i, j]
            if np.isfinite(rho) and abs(rho) >= 0.75:
                graph.add_edge(a, b, weight=abs(rho), rho=rho)
    positions = nx.circular_layout(graph)
    label_positions = {name: xy * 1.17 for name, xy in positions.items()}
    fig, ax = plt.subplots(figsize=(14, 10), layout="constrained")
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=430,
        node_color=[GROUP_COLORS[features.grupo_de(k)] for k in graph],
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        width=[2 * graph[a][b]["weight"] for a, b in graph.edges],
        edge_color=["#307ea4" if graph[a][b]["rho"] > 0 else "#d76d3b" for a, b in graph.edges],
        alpha=0.45,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        label_positions,
        font_size=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
        ax=ax,
    )
    for group, color in GROUP_COLORS.items():
        ax.scatter([], [], color=color, label=group, s=55)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    ax.set(title="Grafo de correlaciones · |ρ| ≥ 0.75 · train")
    ax.set_xlim(-1.48, 1.48)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    doc.figure(
        "correlation_graph",
        fig,
        "Color del nodo: grupo de features. Arista azul: positiva; naranja: negativa. "
        "No implica causalidad.",
    )

    shown = [
        "zero_frac_all",
        "b_3k4_4k",
        "onset0",
        "lat_med",
        "f0_delta_mediana",
        "shimmer_local",
        "razon_jitter_local",
        "razon_shimmer_local",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for ax, name in zip(axes.flat, shown, strict=True):
        j = names.index(name)
        finite = X[np.isfinite(X[:, j]), j]
        lo, hi = np.quantile(finite, [0.01, 0.99])
        bins = np.linspace(lo, hi, 25)
        for target, (label, color) in enumerate(COLORS.items()):
            values = X[(y == target) & np.isfinite(X[:, j]), j]
            ax.hist(values, bins=bins, density=True, alpha=0.55, color=color, label=label)
        ax.set(title=name, ylabel="Densidad")
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=9)
    doc.figure(
        "feature_distributions",
        fig,
        "Distribuciones por clase en train. Vista entre percentiles 1 y 99; "
        "las colas completas se auditan aparte.",
    )


def model_plots(doc, report, evaluations, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for budget, color in zip(features.Presupuesto, ["#087e8b", "#da6c37", "#8064a2"], strict=True):
        rows = [r for r in predictions if r["budget"] == budget and r["split"] == "train_oof"]
        y, p = np.array([r["label"] for r in rows]), np.array([r["p"] for r in rows])
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(
            fpr, tpr, color=color, label=f"{BUDGET_NAMES[budget]} · {roc_auc_score(y, p):.4f}"
        )
        observed, predicted = calibration_curve(y, p, n_bins=6, strategy="quantile")
        axes[1].plot(predicted, observed, "o-", color=color, label=BUDGET_NAMES[budget])
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="#8c9ba3", lw=1)
        ax.legend(fontsize=8)
    axes[0].set(
        title="ROC · predicciones fuera de muestra",
        xlabel="Tasa de falsos positivos",
        ylabel="Sensibilidad",
    )
    axes[1].set(
        title="Calibración Platt · train OOF",
        xlabel="Probabilidad media de sintético",
        ylabel="Fracción sintética observada",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    doc.figure(
        "roc_calibration",
        fig,
        "La calibración y la fusión se reajustan dentro de cada fold; "
        "seis bins de frecuencia similar.",
    )

    for budget in features.Presupuesto:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7.8), layout="constrained")
        for i, split in enumerate(("cv", "val")):
            m = (
                report["budgets"][budget]["cv"]["clean"]
                if split == "cv"
                else report["budgets"][budget]["val"]
            )
            for j, threshold in enumerate(("0.5", "0.7")):
                matrix = np.array(m["thresholds"][threshold]["confusion_matrix"])
                ax = axes[i, j]
                ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
                for (a, b), value in np.ndenumerate(matrix):
                    ax.text(
                        b,
                        a,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="white" if value > matrix.max() / 2 else "#193e55",
                    )
                ax.set(
                    xticks=[0, 1],
                    yticks=[0, 1],
                    xticklabels=["Humana", "Sintética"],
                    yticklabels=["Humana", "Sintética"],
                    xlabel="Predicción",
                    ylabel="Real",
                    title=(
                        f"{'CV por llamada' if split == 'cv' else 'Val final'} · umbral {threshold}"
                    ),
                )
        fig.suptitle(BUDGET_NAMES[budget], fontsize=16)
        doc.figure(
            f"confusion_{budget}",
            fig,
            "Matriz [[TN, FP], [FN, TP]]; positivo = sintético. "
            "Los umbrales se fijaron antes de abrir val.",
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    scenarios = ["clean", "humano_limpio", "bot_evasivo", "bot_ritmo"]
    positions = np.arange(4)
    for j, budget in enumerate(features.Presupuesto):
        result = report["budgets"][budget]["cv"]
        axes[0].bar(
            positions + (j - 1) * 0.24,
            [result[k]["auc"] for k in scenarios],
            0.24,
            label=BUDGET_NAMES[budget],
        )
        axes[1].bar(
            positions + (j - 1) * 0.24,
            [result[k]["thresholds"]["0.5"]["accuracy"] for k in scenarios],
            0.24,
        )
    for ax in axes:
        ax.set(
            xticks=positions,
            xticklabels=["Limpio", "Humano filtrado", "Bot con ruido", "Bot / ritmo"],
            ylim=(0, 1.06),
        )
        ax.tick_params(axis="x", rotation=18)
    axes[0].set(ylabel="ROC-AUC", title="Discriminación bajo estrés · train OOF")
    axes[1].set(ylabel="Exactitud", title="Decisión a umbral 0.5 bajo estrés")
    axes[0].legend(fontsize=8, loc="lower left")
    doc.figure(
        "stress_comparison",
        fig,
        "Ataques aplicados a las llamadas reservadas del fold. "
        "Bot/ritmo modifica solo conducta, no el audio.",
    )


def run(output):
    inputs = joblib.load(output / "report_inputs.joblib")
    records, report = inputs["records"], inputs["report"]
    train = [r for r in records if r["meta"]["split"] == "train"]
    names = list(feature_names())
    X = np.array([r["samples"]["full"]["call_features"] for r in train])
    y = np.array([int(r["meta"]["label"] == "synthetic") for r in train])
    turn_index = [
        j
        for j, name in enumerate(names)
        if features.grupo_de(name) in TURN_GROUPS and name != "speech0_s"
    ]
    T = np.concatenate([r["samples"]["full"]["turn_features"] for r in train])[:, turn_index]
    turn_names = [names[j] for j in turn_index]
    call_stats = describe_matrix(X, names, "call_train")
    turn_stats = describe_matrix(T, turn_names, "turn_train")
    table_csv(output / "feature_audit.csv", call_stats + turn_stats)
    associations, pairs, pearson, spearman = correlation_analysis(X, y, names, output)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#294456",
            "axes.titlecolor": "#173b50",
        }
    )
    doc = Report(output)
    full = report["budgets"]["full"]
    cv, val = full["cv"]["clean"], full["val"]
    headline = [
        f"Dataset auditado: {len(records)} llamadas; "
        f"{report['dataset']['hours']:.3f} horas reales de WAV; "
        f"{report['dataset']['vad_turns']} intervenciones del llamante con el VAD del servicio.",
        f"Modelo completo: CV agrupada por llamada AUC={cv['auc']:.4f}, "
        f"exactitud={cv['thresholds']['0.5']['accuracy']:.2%}, "
        f"precision={cv['thresholds']['0.5']['precision']:.2%}. "
        f"Sanity check val: AUC={val['auc']:.4f}, "
        f"exactitud={val['thresholds']['0.5']['accuracy']:.2%}.",
        "Se entrenan cuatro regresiones por intervención, se agregan sus scores por llamada "
        "y se añaden dos regresiones sobre conducta y razones entre canales. "
        "Una regresión de fusión consume seis scores y CalibratedClassifierCV ajusta Platt.",
        "El preprocesamiento, los scores para la fusión y la calibración se ajustan con llamadas "
        "separadas dentro de cada fold. La selección compara tres configuraciones con "
        "validación agrupada y estrés; val se informa al final, después de congelar la selección.",
        "El PDF, el HTML y el Markdown reúnen el análisis, las tablas y las figuras. "
        "Los CSV conservan los diagnósticos por variable.",
    ]
    doc.text_page("Dataset y modelo · Issue #5", headline)
    doc.heading("Resultados principales")
    for paragraph in headline[:4]:
        doc.paragraph(paragraph)
    headers = ["Presupuesto", "Llamadas", "AUC", "Exactitud", "Precisión", "Recall", "F1", "Brier"]
    doc.paragraph(
        "Validación de desarrollo: cinco folds con StratifiedGroupKFold por llamada. "
        "Cada llamada aparece una sola vez como prueba. Métricas de decisión a umbral 0.5; "
        "la precisión indica qué proporción de los positivos predichos son sintéticos."
    )
    doc.table(headers, metric_rows(report))
    doc.paragraph(
        "Val final, usado únicamente después de guardar selection_frozen.json y el artefacto:"
    )
    doc.table(headers, metric_rows(report, "val"))
    doc.table(
        ["Presupuesto", "AUC CV · IC95%", "Exactitud CV · IC95%", "AUC val · IC95%"],
        [
            [
                BUDGET_NAMES[b],
                " – ".join(f"{v:.4f}" for v in r["cv_ci95"]["auc"]),
                " – ".join(f"{100 * v:.2f}%" for v in r["cv_ci95"]["accuracy"]),
                " – ".join(f"{v:.4f}" for v in r["val_ci95"]["auc"]),
            ]
            for b, r in report["budgets"].items()
        ],
    )
    doc.paragraph(
        "IC95%: 1,000 remuestreos por grupo. Un intervalo bootstrap [1, 1] al no observar "
        "errores expresa el resultado de remuestrear esta muestra; no garantiza perfección futura. "
        "Los folds de desarrollo también sirven para seleccionar entre tres candidatos."
    )

    doc.heading("Inventario y auditoría previa")
    reference_d = [v for r in records for v in r["audit"]["reference_caller_durations"]]
    vad_d = [v for r in records for v in r["audit"]["vad_caller_durations"]]
    duration_error = np.array([r["audit"]["duration_error_s"] for r in records])
    doc.table(
        ["Split", "Clase", "Llamadas", "Turnos de referencia", "Turnos VAD", "Horas WAV"],
        [
            [
                split,
                label,
                len(rs),
                sum(len(r["audit"]["reference_caller_durations"]) for r in rs),
                sum(len(r["audit"]["vad_caller_durations"]) for r in rs),
                f"{sum(r['audit']['duration_s'] for r in rs) / 3600:.3f}",
            ]
            for split in ("train", "val")
            for label in COLORS
            if (
                rs := [
                    r
                    for r in records
                    if r["meta"]["split"] == split and r["meta"]["label"] == label
                ]
            )
        ],
    )
    doc.paragraph(
        f"Se validaron WAV estéreo PCM de 16 bits a 8 kHz, canales diferentes, llamante "
        "no vacío y duraciones válidas. "
        f"Nulos en manifiesto: {report['manifest']['manifest_nulls']}; "
        f"filas repetidas exactas: {report['manifest']['exact_manifest_duplicates']}; "
        f"llamadas excluidas: {len(report['excluded'])}; segmentos de referencia inválidos: "
        f"{report['dataset']['reference_invalid_turns']}; sin turnos del llamante: "
        f"{report['dataset']['no_caller_calls']}. Se comparó el SHA-256 del PCM para detectar "
        "audios repetidos y evitar que crucen particiones."
    )
    doc.paragraph(
        f"Las {len(reference_d):,} intervenciones de referencia duran {np.mean(reference_d):.3f} s "
        f"de media (el issue citaba 2.56 s). El VAD propio produce {len(vad_d):,}, de "
        f"{np.mean(vad_d):.3f} s de media; son segmentaciones distintas, no muestras adicionales "
        "independientes. Se mantienen todos los turnos válidos, incluidos los cortos."
    )
    doc.paragraph(
        f"La duración WAV menos la declarada va de {duration_error.min():+.2f} a "
        f"{duration_error.max():+.2f} s. El manifiesto expresa segundos enteros y se trata "
        "como una duración aproximada. El extractor usa muestras/sample_rate "
        "para calcular duraciones y recortar intervalos."
    )
    clipping = np.array([r["audit"]["clipping_fraction"] for r in records])
    doc.paragraph(
        f"Clipping en ch0: {int((clipping > 0).sum())} llamadas tienen alguna muestra con "
        f"|PCM|≥32700; {int((clipping > 0.01).sum())} superan el 1% del audio. "
        f"Máximo observado: {clipping.max():.3%}. Se registra como diagnóstico y feature."
    )
    basic_plots(doc, train, records, X, y, names, associations, pearson, spearman)

    doc.heading("Limpieza, faltantes y datos atípicos")
    doc.paragraph(
        f"Esquema de {len(names)} features de llamada y {len(turn_names)} acústicas por turno. "
        "Se retiraron los alias exactos f0_jitter_praat y dur_usada_s; sus equivalentes son "
        "f0_delta_mediana y dur_call. Los ceros físicos (p. ej. silencio digital) se conservan. "
        "Tono no estimable, silencio insuficiente, razones sin denominador o latencias sin "
        "observaciones se representan como faltantes."
    )
    doc.paragraph(
        f"En train se detectaron {int(np.isnan(T).sum()):,} celdas faltantes de "
        f"{T.size:,} en features aplicables de turnos ({np.isnan(T).mean():.2%}), "
        f"y {int(np.isnan(X).sum())} de {X.size:,} en las llamadas completas. "
        "La mediana, los indicadores de ausencia, el escalado y el recorte se aprenden "
        "solo en el subconjunto que entrena cada regresión. No se eliminan personas "
        "ni turnos por parecer extremos o por su clase."
    )
    missing_top = sorted(turn_stats, key=lambda r: r["missing_pct"], reverse=True)[:12]
    doc.table(
        ["Feature por turno", "Nulos", "% nulos", "Grupo"],
        [[r["feature"], r["missing"], f"{r['missing_pct']:.2f}%", r["group"]] for r in missing_top],
    )
    constants = [r["feature"] for r in turn_stats if r["unique"] <= 1]
    doc.paragraph(
        "Features constantes o siempre ausentes en train por turno: "
        + ", ".join(constants)
        + ". En particular, la fuga del agente necesita más de un segundo de audio exclusivo "
        "del agente y no se puede estimar con el contexto de 300 ms. Estas columnas se "
        "conservan en el esquema auditable, pero su valor constante después de imputar y "
        "escalar no aporta señal a la regresión."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    missing_names = [r["feature"] for r in missing_top]
    missing_index = [turn_names.index(k) for k in missing_names]
    per_call = np.array(
        [
            np.isnan(r["samples"]["full"]["turn_features"][:, turn_index][:, missing_index]).mean(0)
            for r in train
        ]
    )
    im = axes[0].imshow(per_call[np.argsort(y)].T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set(
        yticks=range(len(missing_names)),
        yticklabels=missing_names,
        xlabel="Llamadas de train: humanas primero, sintéticas después",
        title="Fracción de turnos sin medición por llamada",
    )
    axes[0].axvline(int((y == 0).sum()) - 0.5, color="#223847", lw=1)
    fig.colorbar(im, ax=axes[0], shrink=0.6)
    outlier_top = sorted(call_stats, key=lambda r: r["iqr_outlier_pct"], reverse=True)[:12]
    axes[1].barh(
        [r["feature"] for r in outlier_top][::-1],
        [r["iqr_outlier_pct"] for r in outlier_top][::-1],
        color="#da6c37",
    )
    axes[1].set(
        xlabel="% de observaciones fuera de Q1−1.5·IQR / Q3+1.5·IQR",
        title="Colas extremas · features de llamada en train",
    )
    doc.figure(
        "missing_outliers",
        fig,
        "IQR es una señal de inspección. Columnas de IQR cero se marcan aparte, "
        "sin declarar todos sus valores atípicos.",
    )
    doc.table(
        ["Feature", "% extremos IQR", "Mediana", "P99.5", "Máximo"],
        [
            [
                r["feature"],
                f"{r['iqr_outlier_pct']:.2f}%",
                f"{r['median']:.4g}",
                f"{r['p995']:.4g}",
                f"{r['max']:.4g}",
            ]
            for r in outlier_top[:8]
        ],
    )
    doc.paragraph(
        "La comparación sin recorte conserva la misma imputación y escalado. "
        "La variante con recorte limita cada variable a P0.5–P99.5 de su fold de entrenamiento; "
        "la siguiente tabla mide el efecto manteniendo C=0.33."
    )
    doc.table(
        [
            "Presupuesto",
            "AUC sin recorte",
            "AUC con recorte",
            "Peor AUC estrés sin / con",
            "Elegido",
        ],
        [
            [
                BUDGET_NAMES[b],
                f"{r['sin_recorte_C033']['clean']['auc']:.4f}",
                f"{r['limpio_C033']['clean']['auc']:.4f}",
                " / ".join(
                    f"{min(r[n][s]['auc'] for s in STRESS_NAMES):.4f}"
                    for n in ("sin_recorte_C033", "limpio_C033")
                ),
                report["budgets"][b]["selected"],
            ]
            for b, r in report["candidates"].items()
        ],
    )

    doc.heading("Correlaciones encontradas")
    doc.paragraph(
        "Se calculan Pearson y Spearman por pares completos usando únicamente train. "
        "La asociación con la etiqueta se calcula por llamada, con corrección exploratoria "
        "Benjamini–Hochberg de las pruebas por feature. ρ positivo indica valores mayores "
        "en llamadas sintéticas. Correlación, coeficiente del clasificador y causalidad "
        "son conceptos distintos."
    )
    doc.table(
        ["Feature", "Grupo", "ρ con sintético", "Mediana humana", "Mediana sintética", "q BH"],
        [
            [
                r["feature"],
                r["group"],
                f"{r['spearman_label']:+.3f}",
                f"{r['median_human']:.4g}",
                f"{r['median_synthetic']:.4g}",
                f"{r['q_bh']:.2g}",
            ]
            for r in associations[:12]
        ],
    )
    for group in LAYER_NAMES:
        best = next((r for r in associations if r["group"] == group), None)
        if best:
            doc.paragraph(
                f"{group}: {best['feature']} es la variable del grupo más asociada con "
                f"la etiqueta (ρ={best['spearman_label']:+.3f}); mediana humana "
                f"{best['median_human']:.4g}, sintética {best['median_synthetic']:.4g}."
            )
    doc.table(
        ["Par", "ρ total", "ρ humanas", "ρ sintéticas"],
        [
            [
                r["feature_a"] + " ↔ " + r["feature_b"],
                f"{r['spearman']:+.3f}",
                f"{r['rho_human']:+.3f}",
                f"{r['rho_synthetic']:+.3f}",
            ]
            for r in pairs[:10]
        ],
    )
    doc.paragraph(
        "Las correlaciones dentro de cada clase ayudan a distinguir redundancia acústica "
        "de asociaciones que aparecen al mezclar humanas y sintéticas. La regularización "
        "L2 distribuye el peso entre variables correlacionadas; los mapas no se usaron "
        "para escoger features mirando val. Una señal de ganancia, codec o silencio puede "
        "cambiar con el procesamiento de la llamada; el banco de estrés mide ese efecto."
    )

    doc.heading("Modelo, calibración y regla de acuerdo")
    doc.paragraph(
        "Unidad acústica: un turno del llamante con hasta 300 ms de contexto anterior, "
        "sin leer el siguiente turno. Las regresiones por turno usan todos los turnos "
        "de las llamadas de entrenamiento incluso al aprender la fusión de primera intervención. "
        "El peso total de cada llamada en la pérdida acústica es igual, independientemente "
        "del número de turnos. Cada presupuesto tiene su propia fusión y calibrador."
    )
    doc.paragraph(
        "Agregación de cada capa acústica: 0.6·sigmoid(media de logits) + 0.2·máximo "
        "de probabilidades + 0.2·fracción de turnos con score≥0.7. Los pesos se fijaron "
        "antes de evaluar. Se fusionan esos cuatro scores más conducta y razones entre "
        "canales. StandardScaler + LogisticRegression L2 se usa en cada regresión. "
        "En sklearn 1.9, l1_ratio=0 expresa L2 y evita el argumento penalty deprecado."
    )
    doc.paragraph(
        "La fusión aprende con predicciones de capas fuera de muestra en tres folds "
        "internos. CalibratedClassifierCV vuelve a separar toda la jerarquía en otros "
        "tres folds para ajustar Platt. Se usa sigmoide por el tamaño de muestra; la "
        "documentación de sklearn desaconseja isotónica con pocas muestras de calibración."
    )
    doc.table(
        ["Presupuesto", "Brier antes / Platt", "Log-loss antes / Platt", "ECE Platt"],
        [
            [
                BUDGET_NAMES[b],
                f"{r['cv']['uncalibrated_clean']['brier']:.4f} / {r['cv']['clean']['brier']:.4f}",
                f"{r['cv']['uncalibrated_clean']['log_loss']:.4f} / "
                f"{r['cv']['clean']['log_loss']:.4f}",
                f"{r['cv']['clean']['ece_10_bins']:.4f}",
            ]
            for b, r in report["budgets"].items()
        ],
    )
    doc.paragraph(
        "probability_synthetic conserva P(sintético) calibrada. confidence expresa confianza "
        "en la clase elegida: max(p, 1−p), con tope 0.65 si se predice sintético y menos "
        "de dos capas disponibles votan con score≥0.7; una sola capa disponible no supera "
        "0.9. Este ajuste conservador de confianza se distingue de la probabilidad calibrada; "
        "no cambia las matrices a umbrales 0.5/0.7, que se calculan sobre P(sintético)."
    )
    model_plots(doc, report, inputs["evaluations"], inputs["predictions"])

    doc.heading("Pruebas de estrés y errores")
    doc.paragraph(
        "Humano filtrado: paso bajo 3.4 kHz, puerta de ruido digital y normalización "
        "del pico. Bot con ruido: ruido rosa a −56 dBFS y variaciones de ganancia de ±3 dB "
        "por turno. Ambos se vuelven a segmentar con el VAD. Bot/ritmo adelanta entradas "
        "y añade variabilidad e interrupciones en las features de conducta; es una "
        "ablación de metadatos con el audio intacto. Cada escenario conserva ambas clases "
        "y transforma únicamente la clase objetivo."
    )
    doc.table(
        ["Presupuesto", "Escenario", "AUC", "Exactitud", "Precisión", "Recall", "FP", "FN"],
        [
            [
                BUDGET_NAMES[b],
                scenario,
                f"{m['auc']:.4f}",
                f"{100 * t['accuracy']:.2f}%",
                f"{100 * t['precision']:.2f}%",
                f"{100 * t['recall']:.2f}%",
                t["confusion_matrix"][0][1],
                t["confusion_matrix"][1][0],
            ]
            for b, result in report["budgets"].items()
            for scenario in ("clean", "humano_limpio", "bot_evasivo", "bot_ritmo")
            if (m := result["cv"][scenario]) and (t := m["thresholds"]["0.5"])
        ],
    )
    worst = min(
        (r["cv"][s]["auc"], b, s)
        for b, r in report["budgets"].items()
        for s in ("humano_limpio", "bot_evasivo", "bot_ritmo")
    )
    doc.paragraph(
        f"El peor escenario medido es {worst[2]} con {BUDGET_NAMES[worst[1]].lower()}: "
        f"AUC={worst[0]:.4f}. Un AUC alto del conjunto limpio no elimina los errores "
        "por cambio de canal o conducta. Las matrices y tasas de error permiten ver "
        "si el deterioro afecta más a humanos marcados como bot o a bots que pasan por humanos."
    )
    errors = [r for r in inputs["predictions"] if (r["p"] >= 0.5) != bool(r["label"])]
    if errors:
        table_csv(output / "errors.csv", errors)
        doc.paragraph(
            f"Se exportaron {len(errors)} errores entre los tres presupuestos y "
            "train OOF/val a errors.csv. El mismo audio puede aparecer en varios "
            "presupuestos; no representa ese número de llamadas distintas."
        )
    doc.paragraph(
        "Se guardaron model/model.joblib, métricas por umbral y escenario, predicciones "
        "por llamada, fronteras de las intervenciones, auditoría del audio, estadísticos "
        "por variable y las matrices completas de correlación. Los originales del dataset "
        "no se modificaron. El artefacto y los resultados derivados permanecen locales."
    )
    doc.finish()
    print(f"Reporte: {output / 'reporte.html'}; {output / 'reporte.pdf'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analysis/issue5"))
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
        run(args.output)


if __name__ == "__main__":
    main()
