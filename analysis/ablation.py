"""Ablacion por grupos: separa señal ROBUSTA de atajos fragiles del pipeline.

Un AUC de 1.000 en val no prueba que el modelo entienda "humano vs bot": puede
estar leyendo el codec, la ganancia o el piso de ruido. Aqui se entrena con y sin
cada grupo para ver que queda cuando se le quitan los atajos.

Tambien perfila los falsos positivos: que humanos se marcan como bot y por que.
"""

import csv
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "analysis")

# grupos por significado fisico. El orden importa: el primero que matchea, gana.
GROUPS = [
    ("ganancia",  r"^(rms_speech_db|peak_db|crest_db|rms_noise_db|noise_std_db|"
                  r"leak_rms_db|leak_vs_noise|ag_rms_db|dc_offset|clip_frac|speech0_s)$"),
    ("codec_bw",  r"^(b_\d|b_3k4|roll85|cen_|bw_mean|flat_|zcr_|flux_|ag_cen)"),
    ("silencio",  r"^(sil_|zero_frac|nuniq_sil|noise_rms_cv|xcorr)"),
    ("prosodia",  r"^(f0_|ac_peak|voiced_frac|mod_|env_)"),
]
BEHAV = "conducta"  # todo lo que viene de turns/*.json


def load(name):
    p = os.path.join(A, name)
    if not os.path.exists(p):
        return {}, []
    rows = list(csv.DictReader(open(p)))
    keys = [k for k in rows[0] if k not in ("anon_id", "label", "split")]
    return {r["anon_id"]: r for r in rows}, keys


def group_of(k, from_turns):
    if from_turns:
        return BEHAV
    for g, rx in GROUPS:
        if re.match(rx, k):
            return g
    return "otras_audio"


def auc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(X, y, l2=3.0, iters=4000, lr=0.3):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:] / len(y)
        w -= lr * g
    return w


def pred(w, X):
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))


def main():
    td, tk = load("turn_features.csv")
    ad, ak = load("audio_features.csv")
    ids = sorted(set(td) & set(ad))
    keys = tk + ak
    groups = [group_of(k, True) for k in tk] + [group_of(k, False) for k in ak]
    X = np.array([[float(td[i][k]) for k in tk] + [float(ad[i][k]) for k in ak] for i in ids])
    y = np.array([1 if td[i]["label"] == "synthetic" else 0 for i in ids])
    sp = np.array([td[i]["split"] for i in ids])
    groups = np.array(groups)

    tr, va = sp == "train", sp == "val"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = np.clip((X - mu) / sd, -6, 6)

    uniq = sorted(set(groups))
    print(f"{len(ids)} llamadas, {len(keys)} features en {len(uniq)} grupos")
    for g in uniq:
        print(f"  {g:<14}{int((groups==g).sum()):>4} features")

    def ev(mask, name):
        if mask.sum() == 0:
            return
        w = fit(Z[tr][:, mask], y[tr])
        a_tr = auc(pred(w, Z[tr][:, mask]), y[tr])
        a_va = auc(pred(w, Z[va][:, mask]), y[va])
        p = pred(w, Z[va][:, mask])
        fp = int(((p >= 0.5) & (y[va] == 0)).sum())
        fn = int(((p < 0.5) & (y[va] == 1)).sum())
        print(f"{name:<34}{int(mask.sum()):>5}{a_tr:>10.4f}{a_va:>10.4f}{fp:>6}{fn:>6}")
        return a_va

    print(f"\n{'conjunto':<34}{'n':>5}{'AUC tr':>10}{'AUC val':>10}{'FP':>6}{'FN':>6}")
    print("-" * 71)
    ev(np.ones(len(keys), bool), "TODO")
    for g in uniq:
        ev(groups == g, f"solo {g}")
    print()
    for g in uniq:
        ev(groups != g, f"TODO menos {g}")
    print()
    frag = np.isin(groups, ["ganancia", "codec_bw", "silencio"])
    ev(~frag, "ROBUSTO (conducta+prosodia)")
    ev(np.isin(groups, [BEHAV, "prosodia"]), "conducta + prosodia")
    ev(groups == BEHAV, "solo conducta (sin tocar audio)")

    # ---- perfil de falsos positivos con el modelo robusto ----
    mask = ~frag
    w = fit(Z[tr][:, mask], y[tr])
    p_all = pred(w, Z[:, mask])
    print("\n=== humanos con score mas alto (candidatos a falso positivo) ===")
    hi = [(p_all[i], ids[i], sp[i]) for i in range(len(ids)) if y[i] == 0]
    hi.sort(reverse=True)
    kk = [k for k, m in zip(keys, mask) if m]
    show = ["onset0", "frac_lat_slow", "lat_cv", "lat_med", "n_barge0", "frac_short0",
            "silence_ratio", "f0_jitter"]
    idx = {k: kk.index(k) for k in show if k in kk}
    print(f"{'anon_id':<22}{'split':<7}{'score':>7}" + "".join(f"{k[:11]:>12}" for k in idx))
    for s, i, spl in hi[:8]:
        r = ids.index(i)
        print(f"{i:<22}{spl:<7}{s:>7.3f}" + "".join(f"{X[r][keys.index(k)]:>12.2f}" for k in idx))
    print("\n=== bots con score mas bajo (candidatos a falso negativo) ===")
    lo = [(p_all[i], ids[i], sp[i]) for i in range(len(ids)) if y[i] == 1]
    lo.sort()
    for s, i, spl in lo[:8]:
        r = ids.index(i)
        print(f"{i:<22}{spl:<7}{s:>7.3f}" + "".join(f"{X[r][keys.index(k)]:>12.2f}" for k in idx))
    print("\nmedia humanos:" + " " * 21 + "".join(
        f"{X[(y==0)][:, keys.index(k)].mean():>12.2f}" for k in idx))
    print("media bots:   " + " " * 21 + "".join(
        f"{X[(y==1)][:, keys.index(k)].mean():>12.2f}" for k in idx))


if __name__ == "__main__":
    main()
