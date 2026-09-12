"""Extractor unico de features, con presupuesto de audio.

Un solo punto de entrada para que el entrenamiento y el endpoint usen exactamente el mismo
codigo. Sustituye a analysis/audio_probe.py y analysis/turns_probe.py, que calculaban lo
mismo por separado y con implementaciones propias de cosas que ya estan en librerias.

El presupuesto dice cuanto audio se mira, que es lo que permite la cascada de #6:

    "first_turn"  hasta el final de la primera intervencion del llamante
    "20s"         los primeros 20 s de llamada
    "full"        todo

Las claves devueltas son SIEMPRE las mismas, con 0.0 en lo que no se puede calcular con el
audio disponible. Cada presupuesto entrena su propio modelo: mezclar vectores de
presupuestos distintos en un mismo modelo seria el desajuste de dominio que se quiere evitar.

Cada feature esta etiquetada con su grupo en GRUPOS, para que la ablacion y el banco de
estres no dependan de expresiones regulares sobre los nombres:

    ganancia     a que volumen se grabo            (fragil: cambia con el telefono)
    codec_bw     que cadena de audio lo produjo    (fragil)
    silencio     si hay una sala detras del micro  (fragil)
    prosodia     como suena la voz                 (robusta)
    conducta     como se coordina con el agente    (robusta)
    razon_canal  ch0 comparado con ch1             (robusta por construccion)
"""

import numpy as np
import parselmouth
from parselmouth.praat import call as praat

from app import config, intervalos

EPS = 1e-12
Presupuesto = ("first_turn", "20s", "full")

# --------------------------------------------------------------------------- grupos


def _grupos_por_prefijo():
    """nombre -> grupo. El orden importa: el primer prefijo que encaja, gana."""
    return (
        ("razon_", "razon_canal"),
        ("sil_", "silencio"),
        ("zero_frac", "silencio"),
        ("nuniq_sil", "silencio"),
        ("noise_rms_cv", "silencio"),
        ("xcorr", "silencio"),
        ("rms_speech_db", "ganancia"),
        ("rms_noise_db", "ganancia"),
        ("noise_std_db", "ganancia"),
        ("snr_db", "ganancia"),
        ("peak_db", "ganancia"),
        ("crest_db", "ganancia"),
        ("dc_offset", "ganancia"),
        ("clip_frac", "ganancia"),
        ("leak_", "ganancia"),
        ("ag_rms_db", "ganancia"),
        ("speech0_s", "ganancia"),
        ("b_", "codec_bw"),
        ("roll85", "codec_bw"),
        ("cen_", "codec_bw"),
        ("bw_mean", "codec_bw"),
        ("flat_", "codec_bw"),
        ("zcr_", "codec_bw"),
        ("flux_", "codec_bw"),
        ("ag_cen", "codec_bw"),
        ("f0_", "prosodia"),
        ("jitter", "prosodia"),
        ("shimmer", "prosodia"),
        ("hnr", "prosodia"),
        ("ac_peak", "prosodia"),
        ("voiced_frac", "prosodia"),
        ("mod_", "prosodia"),
        ("env_", "prosodia"),
    )


def grupo_de(nombre: str) -> str:
    """Grupo de una feature. Todo lo que no sea acustico viene de los turnos."""
    for prefijo, grupo in _grupos_por_prefijo():
        if nombre.startswith(prefijo):
            return grupo
    return "conducta"


# ------------------------------------------------------------------------- utilidades


def db(v: float) -> float:
    return 20.0 * np.log10(max(float(v), EPS))


def _marcos(señal: np.ndarray, n: int = 256, salto: int = 128) -> np.ndarray:
    """Ventanas solapadas como matriz (n_marcos, n). Vacia si no cabe ninguna."""
    if len(señal) < n:
        return np.zeros((0, n))
    k = 1 + (len(señal) - n) // salto
    idx = np.arange(n)[None, :] + salto * np.arange(k)[:, None]
    return señal[idx]


def _mascara(turnos: list[dict], canal: int, n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for t in turnos:
        if t["channel"] == canal:
            a = max(0, int(t["start"] * sr))
            b = min(n, int(t["end"] * sr))
            if b > a:
                m[a:b] = True
    return m


def _segmentos(turnos: list[dict], canal: int) -> list[tuple[float, float]]:
    return intervalos.tramos(
        intervalos.union([(t["start"], t["end"]) for t in turnos if t["channel"] == canal])
    )


# ------------------------------------------------------------------ espectro y niveles


def _espectral(señal: np.ndarray, sr: int, prefijo: str = "") -> dict:
    """Estadisticos espectrales sobre marcos de 32 ms."""
    F = _marcos(señal)
    if len(F) == 0:
        return {}
    ventana = np.hanning(F.shape[1])
    S = np.abs(np.fft.rfft(F * ventana, axis=1)) ** 2
    frecs = np.fft.rfftfreq(F.shape[1], 1 / sr)
    total = S.sum(axis=1) + EPS

    centroide = (S * frecs).sum(axis=1) / total
    ancho = np.sqrt((S * (frecs[None, :] - centroide[:, None]) ** 2).sum(axis=1) / total)
    acumulado = np.cumsum(S, axis=1) / total[:, None]
    rolloff = frecs[np.argmax(acumulado >= 0.85, axis=1)]
    planitud = np.exp(np.mean(np.log(S + EPS), axis=1)) / (S.mean(axis=1) + EPS)

    def banda(lo, hi):
        sel = (frecs >= lo) & (frecs < hi)
        return (S[:, sel].sum(axis=1) / total).mean()

    Sn = S / total[:, None]
    flujo = np.sqrt(((Sn[1:] - Sn[:-1]) ** 2).sum(axis=1)) if len(Sn) > 1 else np.zeros(0)

    f = {
        "cen_mean": centroide.mean(), "cen_std": centroide.std(),
        "bw_mean": ancho.mean(),
        "roll85_mean": rolloff.mean(), "roll85_std": rolloff.std(),
        "flat_mean": planitud.mean(), "flat_std": planitud.std(),
        "b_0_300": banda(0, 300), "b_300_1k": banda(300, 1000),
        "b_1k_2k": banda(1000, 2000), "b_2k_3k": banda(2000, 3000),
        "b_3k_4k": banda(3000, 4000),
        # por encima del corte telefonico: la firma del remuestreo del TTS
        "b_3k4_4k": banda(3400, 4000),
        "flux_mean": flujo.mean() if len(flujo) else 0.0,
        "flux_std": flujo.std() if len(flujo) else 0.0,
    }
    return {prefijo + k: float(v) for k, v in f.items()}


def _envolvente(señal: np.ndarray, sr: int) -> dict:
    """Cruces por cero, dinamica y espectro de modulacion de la envolvente."""
    F = _marcos(señal)
    if len(F) == 0:
        return {}
    signos = np.sign(F)
    zcr = (np.diff(signos, axis=1) != 0).mean(axis=1)
    rms = np.sqrt((F ** 2).mean(axis=1) + EPS)
    nivel_db = 20 * np.log10(rms / 32768 + EPS)

    f = {
        "zcr_mean": zcr.mean(), "zcr_std": zcr.std(),
        "env_cv": rms.std() / (rms.mean() + EPS),
        "env_range_db": np.percentile(nivel_db, 95) - np.percentile(nivel_db, 5),
        "mod_2_6hz": 0.0, "mod_6_12hz": 0.0, "mod_peak_hz": 0.0,
    }
    # ritmo silabico: el habla humana concentra energia en 2-6 Hz
    envolvente = rms - rms.mean()
    if len(envolvente) > 128:
        M = np.abs(np.fft.rfft(envolvente * np.hanning(len(envolvente))))
        frecs = np.fft.rfftfreq(len(envolvente), 128 / sr)
        total = M.sum() + EPS
        f["mod_2_6hz"] = M[(frecs >= 2) & (frecs < 6)].sum() / total
        f["mod_6_12hz"] = M[(frecs >= 6) & (frecs < 12)].sum() / total
        f["mod_peak_hz"] = frecs[1 + np.argmax(M[1:])]
    return {k: float(v) for k, v in f.items()}


# ---------------------------------------------------------------------------- prosodia


def _prosodia(x: np.ndarray, sr: int, segmentos: list[tuple[float, float]]) -> dict:
    """F0, jitter, shimmer y HNR con Praat, segmento a segmento.

    Se calcula por segmento y no sobre el habla concatenada: pegar trozos no contiguos
    crea discontinuidades artificiales, y jitter y shimmer se miden sobre periodos
    glotales consecutivos. El estimador propio por autocorrelacion que habia antes daba
    jitter MAYOR en los bots, al revez de la literatura; ver la nota al final del modulo.
    """
    vacio = {
        "f0_mean": 0.0, "f0_std": 0.0, "f0_cv": 0.0, "f0_p5": 0.0, "f0_p95": 0.0,
        "f0_range": 0.0, "f0_jitter_praat": 0.0, "f0_delta_mediana": 0.0,
        "f0_smooth": 0.0, "voiced_frac": 0.0, "ac_peak_mean": 0.0, "ac_peak_std": 0.0,
        "jitter_local": 0.0, "shimmer_local": 0.0, "hnr_db": 0.0,
    }
    f0_todos, fuerza_todos, saltos = [], [], []
    jitter, shimmer, hnr = [], [], []
    n_tramas = 0

    for inicio, fin in segmentos:
        a, b = int(inicio * sr), min(len(x), int(fin * sr))
        if b - a < int(0.25 * sr):  # Praat necesita algo de senal para un piso de 60 Hz
            continue
        trozo = x[a:b].astype(np.float64) / 32768.0
        sonido = parselmouth.Sound(trozo, sampling_frequency=sr)
        try:
            tono = sonido.to_pitch_ac(time_step=0.02, pitch_floor=60.0, pitch_ceiling=350.0)
        except Exception:
            continue
        f0 = tono.selected_array["frequency"]
        fuerza = tono.selected_array["strength"]
        n_tramas += len(f0)
        con_voz = f0 > 0
        if con_voz.sum() > 3:
            f0_todos.append(f0[con_voz])
            fuerza_todos.append(fuerza[con_voz])
            # micro-variacion sobre tramas de voz CONSECUTIVAS, no a traves de huecos
            bordes = np.flatnonzero(np.diff(con_voz.astype(np.int8)) != 0) + 1
            for racha in np.split(f0, bordes):
                if racha[0] > 0 and len(racha) > 2:
                    saltos.append(np.abs(np.diff(racha)) / racha[:-1])
        try:
            puntos = praat(sonido, "To PointProcess (periodic, cc)", 60.0, 350.0)
            jitter.append(praat(puntos, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
            shimmer.append(praat([sonido, puntos], "Get shimmer (local)",
                                 0, 0, 1e-4, 0.02, 1.3, 1.6))
            hnr.append(praat(sonido.to_harmonicity_cc(minimum_pitch=60.0), "Get mean", 0, 0))
        except Exception:
            pass

    if not f0_todos:
        return vacio

    f0 = np.concatenate(f0_todos)
    fuerza = np.concatenate(fuerza_todos)
    d = np.concatenate(saltos) if saltos else np.zeros(1)

    def limpio(valores):
        v = np.array([x for x in valores if np.isfinite(x)])
        return float(v.mean()) if len(v) else 0.0

    f = dict(vacio)
    f.update({
        "f0_mean": f0.mean(), "f0_std": f0.std(), "f0_cv": f0.std() / (f0.mean() + EPS),
        "f0_p5": np.percentile(f0, 5), "f0_p95": np.percentile(f0, 95),
        "f0_range": np.percentile(f0, 95) - np.percentile(f0, 5),
        "f0_jitter_praat": np.median(d),        # micro-variacion relativa entre tramas
        "f0_delta_mediana": np.median(d),
        "f0_smooth": float((d < 0.02).mean()),  # contornos demasiado suaves
        "voiced_frac": len(f0) / max(1, n_tramas),
        "ac_peak_mean": fuerza.mean(), "ac_peak_std": fuerza.std(),
        "jitter_local": limpio(jitter), "shimmer_local": limpio(shimmer), "hnr_db": limpio(hnr),
    })
    return {k: float(v) for k, v in f.items()}


# ---------------------------------------------------------------------------- acustica


def _acustica(x: np.ndarray, sr: int, turnos: list[dict]) -> dict:
    """Features del canal del llamante, separando habla de silencio con los turnos."""
    n = len(x)
    c0 = x[:, 0].astype(np.float64)
    c1 = x[:, 1].astype(np.float64)

    habla0 = _mascara(turnos, 0, n, sr)
    habla1 = _mascara(turnos, 1, n, sr)
    silencio = ~habla0 & ~habla1  # silencio real de los dos canales

    s0, z0 = c0[habla0], c0[silencio]
    f: dict[str, float] = {}

    # --- niveles y ruido ---
    f["rms_speech_db"] = db(np.sqrt((s0 ** 2).mean()) / 32768) if len(s0) else 0.0
    f["rms_noise_db"] = db(np.sqrt((z0 ** 2).mean()) / 32768) if len(z0) else 0.0
    f["snr_db"] = f["rms_speech_db"] - f["rms_noise_db"]
    f["peak_db"] = db(np.abs(s0).max() / 32768) if len(s0) else 0.0
    f["crest_db"] = f["peak_db"] - f["rms_speech_db"]
    f["dc_offset"] = float(c0.mean() / 32768)
    f["clip_frac"] = float((np.abs(s0) >= 32700).mean()) if len(s0) else 0.0

    # --- silencio digital: el TTS y la mezcla sintetica dejan ceros exactos ---
    f["zero_frac_sil"] = float((z0 == 0).mean()) if len(z0) else 0.0
    f["zero_frac_all"] = float((c0 == 0).mean())
    f["nuniq_sil"] = len(np.unique(z0)) / max(len(z0), 1)
    f["noise_std_db"] = db(z0.std() / 32768) if len(z0) else 0.0
    marcos_z = _marcos(z0)
    if len(marcos_z):
        rms_z = np.sqrt((marcos_z ** 2).mean(axis=1) + EPS)
        f["noise_rms_cv"] = float(rms_z.std() / (rms_z.mean() + EPS))
    else:
        f["noise_rms_cv"] = 0.0

    f.update(_espectral(z0, sr, prefijo="sil_"))
    f.update(_espectral(s0, sr))
    f.update(_envolvente(s0, sr))
    f.update(_prosodia(x[:, 0], sr, _segmentos(turnos, 0)))

    # --- fuga del agente en el canal del llamante ---
    solo_agente = habla1 & ~habla0
    if solo_agente.sum() > sr:
        fuga = c0[solo_agente]
        f["leak_rms_db"] = db(np.sqrt((fuga ** 2).mean()) / 32768)
        f["leak_vs_noise"] = f["leak_rms_db"] - f["rms_noise_db"]
        a, b = c0[solo_agente][:200000], c1[solo_agente][:200000]
        f["xcorr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    else:
        f["leak_rms_db"] = f["leak_vs_noise"] = f["xcorr"] = 0.0

    # --- control: el canal del agente, que el llamante no controla ---
    s1 = c1[habla1]
    f["ag_rms_db"] = db(np.sqrt((s1 ** 2).mean()) / 32768) if len(s1) > sr else 0.0
    f["ag_cen"] = _espectral(s1, sr).get("cen_mean", 0.0) if len(s1) > sr else 0.0
    return f


# ---------------------------------------------------------------------------- conducta


def conducta(turnos: list[dict], duracion: float) -> dict:
    """Features de conducta a partir de los turnos, sin tocar el audio.

    Publica porque hay scripts que solo necesitan esta capa y no quieren pagar el coste
    de leer el WAV (scripts/vad_acuerdo.py).
    """
    return _conducta(turnos, duracion)


def _conducta(turnos: list[dict], duracion: float) -> dict:
    """Features de coordinacion, solo a partir de los turnos. Ni un sample de audio."""
    m0, m1 = _segmentos(turnos, 0), _segmentos(turnos, 1)
    bruto0 = [(t["start"], t["end"]) for t in turnos if t["channel"] == 0]
    bruto1 = [(t["start"], t["end"]) for t in turnos if t["channel"] == 1]
    if not m0 or not m1:
        return {}

    u0 = intervalos.union(m0)
    u1 = intervalos.union(m1)
    d0 = [b - a for a, b in bruto0]
    d1 = [b - a for a, b in bruto1]
    habla0 = intervalos.duracion(u0)
    habla1 = intervalos.duracion(u1)

    f = {
        "dur_call": duracion,
        "n_seg0": len(bruto0), "n_seg1": len(bruto1),
        "speech0_s": habla0,
        "speech_ratio0": habla0 / duracion, "speech_ratio1": habla1 / duracion,
        "talk_balance": habla0 / (habla0 + habla1 + EPS),
        "seg_rate0": len(bruto0) / duracion * 60,
        "seg0_mean": float(np.mean(d0)), "seg0_med": float(np.median(d0)),
        "seg0_std": float(np.std(d0)), "seg0_max": max(d0), "seg0_min": min(d0),
        "seg0_p90": float(np.percentile(d0, 90)),
        "frac_short0": sum(1 for d in d0 if d < 0.5) / len(d0),
        "frac_vshort0": sum(1 for d in d0 if d < 0.3) / len(d0),
        "frac_long0": sum(1 for d in d0 if d > 8) / len(d0),
        "seg1_mean": float(np.mean(d1)),
        "seg1_cv": float(np.std(d1) / (np.mean(d1) + EPS)),
    }
    f["seg0_cv"] = f["seg0_std"] / (f["seg0_mean"] + EPS)

    # --- latencia: fin de turno del agente -> inicio del llamante ---
    # Con signo: negativa significa que el llamante se le echo encima. Esa es la version
    # correcta de la senal, porque no depende de cuanto durase el turno del agente.
    latencias, con_signo = [], []
    for inicio0, _ in m0:
        anteriores = [fin for _, fin in m1 if fin <= inicio0 + 1e-9]
        encima = [fin for ini, fin in m1 if ini < inicio0 < fin]
        if encima:
            con_signo.append(inicio0 - encima[0])  # negativa: pisa al agente
        elif anteriores:
            hueco = inicio0 - max(anteriores)
            if 0 <= hueco < 15:
                latencias.append(hueco)
                con_signo.append(hueco)

    f["n_lat"] = len(latencias)
    if latencias:
        lat = np.array(latencias)
        f.update({
            "lat_mean": lat.mean(), "lat_med": float(np.median(lat)), "lat_std": lat.std(),
            "lat_min": lat.min(), "lat_max": lat.max(),
            "lat_mad": float(np.mean(np.abs(lat - np.median(lat)))),
            "frac_lat_fast": float((lat < 0.35).mean()),
            "frac_lat_slow": float((lat > 2.0).mean()),
            "lat_iqr": (float(np.percentile(lat, 75) - np.percentile(lat, 25))
                        if len(lat) > 3 else 0.0),
        })
        f["lat_cv"] = f["lat_std"] / (f["lat_mean"] + EPS)
    else:
        for k in ("lat_mean", "lat_med", "lat_std", "lat_min", "lat_max", "lat_mad",
                  "frac_lat_fast", "frac_lat_slow", "lat_iqr", "lat_cv"):
            f[k] = 0.0

    if con_signo:
        cs = np.array(con_signo)
        f["lat_signo_mean"] = float(cs.mean())
        f["lat_signo_med"] = float(np.median(cs))
        f["frac_lat_negativa"] = float((cs < 0).mean())
    else:
        f["lat_signo_mean"] = f["lat_signo_med"] = f["frac_lat_negativa"] = 0.0

    # --- entrada al saludo: la senal mas limpia del dataset ---
    # 85 de 150 humanos hablan antes del segundo 2; de 203 bots, ninguno. Normalizada
    # contra el fin del primer turno del agente, mide el comportamiento y no la duracion
    # del saludo, que los humanos acortan justamente al pisarlo.
    inicio0, fin0 = m0[0]
    inicio1, fin1 = m1[0]
    f["onset0"] = inicio0
    f["onset1"] = inicio1
    f["lat_primera"] = inicio0 - fin1
    f["pisa_saludo"] = float(inicio0 < fin1)
    f["habla_antes_2s"] = float(inicio0 < 2.0)
    f["dur_saludo"] = fin1 - inicio1

    # --- solapes e interrupciones ---
    solape = intervalos.solape(u0, u1)
    f["overlap_s"] = solape
    f["overlap_ratio0"] = solape / (habla0 + EPS)
    f["overlap_per_min"] = solape / duracion * 60
    f["n_barge0"] = sum(1 for a, _ in m0 if any(i < a < j for i, j in m1))
    f["barge0_rate"] = f["n_barge0"] / len(m0)
    f["n_barge1"] = sum(1 for a, _ in m1 if any(i < a < j for i, j in m0))
    f["barge1_rate"] = f["n_barge1"] / len(m1)

    # --- titubeo: micro-pausas dentro de la misma intervencion ---
    huecos = [bruto0[i + 1][0] - bruto0[i][1] for i in range(len(bruto0) - 1)]
    huecos = [h for h in huecos if 0 < h < 2.0]
    f["n_micropause0"] = len(huecos)
    f["micropause_rate0"] = len(huecos) / len(bruto0)
    f["micropause_mean"] = float(np.mean(huecos)) if huecos else 0.0

    # --- silencios globales ---
    todo = intervalos.union(m0 + m1)
    f["silence_ratio"] = 1 - intervalos.duracion(todo) / duracion
    f["tail_silence"] = duracion - max(fin for _, fin in intervalos.tramos(todo))

    # --- deriva de la latencia: los bots la tienen plana ---
    if len(latencias) >= 6:
        xs = np.arange(len(latencias), dtype=float)
        lat = np.array(latencias)
        f["lat_slope"] = float(np.polyfit(xs, lat, 1)[0])
        mitad = len(lat) // 2
        f["lat_drift"] = float(lat[mitad:].mean() - lat[:mitad].mean())
    else:
        f["lat_slope"] = f["lat_drift"] = 0.0

    return {k: float(v) for k, v in f.items()}


# ------------------------------------------------------------------- razones ch0 / ch1


def _perfil(x: np.ndarray, sr: int, turnos: list[dict], canal: int) -> dict:
    """Retrato corto de un canal, para poder comparar los dos dentro de la misma llamada."""
    segmentos = _segmentos(turnos, canal)
    habla = x[_mascara(turnos, canal, len(x), sr), canal].astype(np.float64)
    if len(habla) < sr // 2:
        return {}
    p = {"nivel_db": db(np.sqrt((habla ** 2).mean()) / 32768),
         "crest_db": db(np.abs(habla).max() / 32768) - db(np.sqrt((habla ** 2).mean()) / 32768)}
    p.update(_espectral(habla, sr))
    p.update(_envolvente(habla, sr))
    p.update(_prosodia(x[:, canal], sr, segmentos))
    return p


def _razones(x: np.ndarray, sr: int, turnos: list[dict]) -> dict:
    """Compara el llamante con el agente DENTRO de la misma grabacion.

    El canal 1 es un TTS conocido en todas las llamadas, asi que sirve de referencia: la
    razon cancela lo que es comun a la grabacion (ganancia, codec, ruido de linea) y deja
    lo que distingue una cadena digital limpia de un microfono en una habitacion. Es la
    forma de usar el canal del agente sin caer en la fuga de mirarlo solo a el.
    """
    claves_resta = ("nivel_db", "crest_db", "hnr_db")
    claves_cociente = ("cen_mean", "b_3k4_4k", "b_3k_4k", "flat_mean", "zcr_mean",
                       "mod_2_6hz", "env_cv", "f0_cv", "jitter_local", "shimmer_local",
                       "ac_peak_mean", "voiced_frac")
    vacio = {f"razon_{k}": 0.0 for k in claves_resta + claves_cociente}

    p0 = _perfil(x, sr, turnos, 0)
    p1 = _perfil(x, sr, turnos, 1)
    if not p0 or not p1:
        return vacio

    f = dict(vacio)
    for k in claves_resta:
        f[f"razon_{k}"] = float(p0.get(k, 0.0) - p1.get(k, 0.0))
    for k in claves_cociente:
        f[f"razon_{k}"] = float(p0.get(k, 0.0) / (abs(p1.get(k, 0.0)) + EPS))
    return f


# -------------------------------------------------------------------------- presupuesto


def _recortar(x: np.ndarray, sr: int, turnos: list[dict], presupuesto: str):
    """Recorta audio y turnos al presupuesto pedido. Devuelve (x, turnos, duracion)."""
    duracion = len(x) / sr
    if presupuesto == "full":
        tope = duracion
    elif presupuesto == "20s":
        tope = min(20.0, duracion)
    elif presupuesto == "first_turn":
        primeros = [t["end"] for t in turnos if t["channel"] == 0]
        tope = min(primeros[0], duracion) if primeros else min(20.0, duracion)
    else:
        raise ValueError(f"presupuesto desconocido: {presupuesto!r}, use uno de {Presupuesto}")

    if tope >= duracion:
        return x, turnos, duracion

    n = max(1, int(tope * sr))
    recortados = []
    for t in turnos:
        if t["start"] >= tope:
            continue
        recortados.append({**t, "end": min(t["end"], tope)})
    return x[:n], recortados, tope


def extraer(x: np.ndarray, sr: int, turnos: list[dict],
            presupuesto: str = "full") -> dict:
    """Vector de features del llamante para el presupuesto de audio indicado.

    x: (n, 2) en int16, canal 0 = llamante. turnos: salida de app.vad.turnos().
    Devuelve siempre las mismas claves; 0.0 donde no hay audio suficiente.
    """
    if sr != config.SAMPLE_RATE:
        raise ValueError(f"se esperaba {config.SAMPLE_RATE} Hz, llego {sr}")
    if x.ndim != 2 or x.shape[1] != config.CHANNELS:
        raise ValueError(f"se esperaba (n, {config.CHANNELS}), llego {x.shape}")

    x, turnos, duracion = _recortar(x, sr, turnos, presupuesto)

    f = {"dur_usada_s": duracion}
    f.update(_conducta(turnos, duracion))
    f.update(_acustica(x, sr, turnos))
    f.update(_razones(x, sr, turnos))
    return {k: (0.0 if v is None or not np.isfinite(v) else float(v)) for k, v in f.items()}


def claves(presupuesto: str = "full") -> list[str]:
    """Claves que devuelve extraer(), en orden estable. Util para fijar el vector."""
    sr = config.SAMPLE_RATE
    x = np.zeros((sr * 30, config.CHANNELS), dtype=np.int16)
    x[sr:sr * 5, 0] = 1000
    x[sr * 6:sr * 10, 1] = 1000
    turnos = [{"channel": 0, "start": 1.0, "end": 5.0},
              {"channel": 1, "start": 6.0, "end": 10.0}]
    return sorted(extraer(x, sr, turnos, presupuesto))


# ------------------------------------------------------------------- nota sobre el jitter
#
# El informe de exploracion daba como sorpresa que el jitter de F0 fuese MAYOR en los bots
# (0.0213 humano contra 0.0259 sintetico), al revez de la literatura. Medido de nuevo con
# Praat sobre las 282 llamadas de train, la sorpresa se deshace y aparece algo mas fino:
#
#   jitter_local (ciclo a ciclo, Praat)   humano 0.0269  bot 0.0274  AUC 0.555  -> NO separa
#   shimmer_local                         humano 0.1322  bot 0.1131  AUC 0.266  -> bot menor
#   f0_jitter_praat (delta entre tramas)  humano 0.0161  bot 0.0208  AUC 0.788  -> bot mayor
#   hnr_db                                humano 10.79   bot 10.82   AUC 0.531  -> NO separa
#
# Es decir: el jitter ciclo a ciclo, medido bien, no separa nada; la "sorpresa" era en buena
# parte el estimador por autocorrelacion propio degradandose sobre audio humano ruidoso a
# 8 kHz. Lo que si separa es el shimmer (bot mas estable, como dice la literatura) y el delta
# de F0 entre tramas, que no es micro-perturbacion sino movimiento del contorno: el TTS
# modula el tono mas de trama a trama y a la vez tiene periodos glotales mas limpios. Las dos
# cosas son ciertas y no se contradicen.
