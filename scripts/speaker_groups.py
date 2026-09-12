"""Deriva identidades de hablante y de voz TTS por agrupamiento, sin etiquetas.

El dataset no da IDs de hablante, asi que agrupar la validacion cruzada solo por llamada
deja que el mismo humano aparezca en varios folds y el resultado sale optimista. Aqui se
derivan las identidades a partir de la voz y se escribe el mapa que consume
scripts/train_issue5.py con --speakers.

Como se hace:

  1. Huella de voz por llamada: MFCC de Praat sobre el habla del llamante (canal 0),
     media y desviacion de los coeficientes 1..12. Se descarta el coeficiente 0, que es
     energia: interesa el timbre, no la ganancia del telefono.
  2. Agrupamiento jerarquico de Ward, por separado en humanos y en sinteticos: una persona
     y una voz de TTS no pueden ser la misma entidad.
  3. Se agrupa DENTRO de cada split. La validacion cruzada solo se hace sobre train, asi
     que nunca hace falta un grupo que cruce train y val; hacerlo por construccion
     satisface la comprobacion de scripts/issue5_data.py.
  4. El umbral se elige por silueta, con un suelo de grupos para que 5 folds sigan siendo
     posibles.

Aviso honesto sobre el limite de esto: se intento calibrar el umbral con la restriccion que
el dataset garantiza (train y val disjuntos por hablante) y NO se pudo. Con la huella MFCC,
incluso agrupando fino (46 grupos para 150 humanos) quedaban 6 grupos mezclando llamadas de
train y de val, que por construccion son personas distintas. Es decir: a 8 kHz esta huella
no identifica personas con fiabilidad. Lo que se obtiene son identidades APROXIMADAS, utiles
para agrupar de mas y hacer la CV mas pesimista, no un reconocimiento de hablante.

Por eso el entrenamiento se corre con las dos agrupaciones y se comparan: si el resultado
apenas se mueve, la fuga por hablante no era material, y eso tambien es un resultado.

uso: python -m scripts.speaker_groups [--salida speaker_groups.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import vad  # noqa: E402

N_COEF = 12
MIN_TRAIN = 15  # para que 5 folds agrupados sigan siendo posibles
MIN_VAL = 3


def huella(ruta: Path) -> np.ndarray | None:
    """Media y desviacion de los MFCC del habla del llamante. None si no hay habla."""
    x, sr = sf.read(ruta, dtype="int16", always_2d=True)
    turnos = [t for t in vad.turnos(x, sr) if t["channel"] == 0]
    if not turnos:
        return None
    trozos = [x[int(t["start"] * sr):int(t["end"] * sr), 0] for t in turnos]
    habla = np.concatenate([t for t in trozos if len(t)])
    if len(habla) < sr:
        return None
    sonido = parselmouth.Sound(habla.astype(np.float64) / 32768.0, sampling_frequency=sr)
    coef = sonido.to_mfcc(number_of_coefficients=N_COEF).to_array()[1:]  # sin el c0 de energia
    if coef.shape[1] < 10:
        return None
    return np.concatenate([coef.mean(axis=1), coef.std(axis=1)])


def agrupar(X: np.ndarray, nombre: str, minimo: int) -> np.ndarray:
    """Umbral por silueta, con suelo de grupos para que la CV agrupada siga siendo viable."""
    Z = StandardScaler().fit_transform(X)
    print(f"\n  {nombre}: {len(Z)} llamadas")
    print(f"    {'umbral':>8}{'grupos':>8}{'mayor':>8}{'silueta':>10}")
    mejor = None
    for umbral in (4, 6, 8, 10, 12, 15, 20, 25, 30, 40):
        etiquetas = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(umbral), linkage="ward"
        ).fit_predict(Z)
        n_grupos = len(set(etiquetas))
        if n_grupos < 2 or n_grupos >= len(Z):
            continue
        sil = silhouette_score(Z, etiquetas)
        viable = n_grupos >= minimo
        print(f"    {umbral:>8}{n_grupos:>8}{max(np.bincount(etiquetas)):>8}{sil:>10.3f}"
              f"  {'' if viable else '(pocos grupos)'}")
        if viable and (mejor is None or sil > mejor[0]):
            mejor = (sil, umbral, etiquetas, n_grupos)
    if mejor is None:
        raise SystemExit(f"ningun umbral deja al menos {minimo} grupos en {nombre}")
    sil, umbral, etiquetas, n_grupos = mejor
    tamanos = np.bincount(etiquetas)
    print(f"    -> umbral {umbral}: {n_grupos} identidades, silueta {sil:.3f}, "
          f"mediana {int(np.median(tamanos))} y maximo {tamanos.max()} llamadas por identidad")
    return etiquetas


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raiz", type=Path, default=ROOT)
    p.add_argument("--salida", type=Path, default=ROOT / "speaker_groups.csv")
    args = p.parse_args()

    filas = list(csv.DictReader(open(args.raiz / "manifest.csv")))
    huellas, usables = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(filas):
        ruta = args.raiz / "audio" / f"{r['anon_id']}.wav"
        if not ruta.exists():
            continue
        h = huella(ruta)
        if h is None:
            print(f"  sin habla utilizable: {r['anon_id']}")
            continue
        huellas.append(h)
        usables.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(filas)}  {time.perf_counter()-t0:.0f}s", flush=True)

    X = np.array(huellas)
    splits = np.array([r["split"] for r in usables])
    etiquetas = np.array([r["label"] for r in usables])
    print(f"\n{len(X)} huellas de {X.shape[1]} dimensiones en {time.perf_counter()-t0:.0f}s")

    identidades = {}
    for clase, columna in (("human", "speaker_id"), ("synthetic", "voice_id")):
        for split in ("train", "val"):
            sel = (etiquetas == clase) & (splits == split)
            minimo = MIN_TRAIN if split == "train" else MIN_VAL
            grupos = agrupar(X[sel], f"{clase} / {split} -> {columna}", minimo)
            filas_sel = [r for r, s in zip(usables, sel, strict=True) if s]
            for r, g in zip(filas_sel, grupos, strict=True):
                identidades[r["anon_id"]] = (columna, f"{columna[0]}{split[0]}{g:03d}")

    faltan = [r["anon_id"] for r in filas if r["anon_id"] not in identidades]
    if faltan:
        raise SystemExit(f"{len(faltan)} llamadas sin identidad; el mapa debe cubrirlas todas")

    with open(args.salida, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["anon_id", "speaker_id", "voice_id"])
        for r in filas:
            columna, valor = identidades[r["anon_id"]]
            w.writerow([r["anon_id"],
                        valor if columna == "speaker_id" else "",
                        valor if columna == "voice_id" else ""])
    print(f"\n-> {args.salida}  ({len(filas)} llamadas)")
    print("   usar con: python -m scripts.train_issue5 --speakers speaker_groups.csv")


if __name__ == "__main__":
    main()
