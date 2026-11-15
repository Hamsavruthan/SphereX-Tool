"""
SPHEREx Spektralklassifikation (0,75-5,0 um) - Referenzprofile, Normierung und Entscheidungslogik.

Passt zu spherex_api.py (v2.1): Eingabe sind Wellenlaenge (um), Flussdichte F_nu (mJy) und
1-sigma-Fehler (mJy), so wie sie /api/load_spherex_mef je Bild liefert
(exact_wavelength_um, aperture_flux_mjy, aperture_flux_err_mjy).

Kennzeichnung der Zahlen in dieser Datei:
  [E] empirisch  : Pecaut & Mamajek (2013, ApJS 208, 9), Tabelle "EEM_dwarf_UBVIJHK_colors_Teff",
                   Version 2022.04.16; Farben mit veroeffentlichten Nullpunkten in F_nu-Verhaeltnisse
                   umgerechnet (siehe color_to_ratio).
  [M] Modell     : Rayleigh-Jeans-Gesetz.
  [V] Vorschlag  : Voreinstellungen und Schnittgrenzen, die an echten SPHEREx-Spektren zu kalibrieren sind.

WICHTIG - Grenzen des Verfahrens:
  * Neutronensterne und Supererden werden NIE als Ergebnis ausgegeben. Beide sind mit SPHEREx im
    Bereich 0,75-5,0 um nicht identifizierbar (zu schwach bzw. nicht vom Stern getrennt).
  * "Gasplanet" wird nur als Spektraltyp "kuehle Methan-Atmosphaere" erkannt (T/Y-Zwerg-Analogon).
    Ob es ein Planet oder ein Brauner Zwerg ist, entscheidet das Spektrum nicht.
  * "Zwerg" (Leuchtkraftklasse V) laesst sich nur mit Parallaxe pruefen (Argument parallax_mas).
  * R = 200: Bedeutung ungeklaert. Hier als Aufloesungsvermoegen lambda/dlambda gefuehrt und nur auf
    REFERENZPROFILE angewandt (r_grid, smooth_to_r). SPHEREx-Messungen haben R = 35-130 und lassen
    sich nicht auf R = 200 bringen.

Nur numpy noetig.
"""
from __future__ import annotations

import math
import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Stuetzfenster und Nullpunkte
# ----------------------------------------------------------------------------------------------
# name: (Referenzwellenlaenge um [E], Mittelungsbereich um [V], Vega-Nullpunkt Jy [E])
WINDOWS = {
    "I":  (0.798, (0.75, 0.88), 2416.0),     # Cousins I   (Bessell et al. 1998)
    "J":  (1.235, (1.17, 1.33), 1594.0),     # 2MASS J     (Cohen et al. 2003)
    "H":  (1.662, (1.49, 1.78), 1024.0),     # 2MASS H
    "Ks": (2.159, (2.03, 2.30), 666.7),      # 2MASS Ks
    "W1": (3.353, (2.90, 3.80), 309.54),     # WISE W1     (Jarrett et al. 2011)
    "W2": (4.603, (4.10, 5.00), 171.787),    # WISE W2; SPHEREx endet bei 5,0 um -> abgeschnitten
}
NORM = "H"
STAR_BANDS = ("I", "J", "Ks", "W1")          # W2 fehlt in der Quelltabelle fuer K7-M4 -> nicht im Sternfit

# Natives Aufloesungsvermoegen der SPHEREx-Baender [E] (Crill et al., arXiv:2404.11017, Tab. 2)
SPHEREX_BANDS = [(0.75, 1.12, 41), (1.10, 1.65, 41), (1.63, 2.44, 41),
                 (2.40, 3.85, 35), (3.81, 4.43, 110), (4.41, 5.01, 130)]


def color_to_ratio(band: str, mag_band_minus_h: float) -> float:
    """Formel 4: q_X = F_nu(X)/F_nu(H) = ZP_X/ZP_H * 10^(-0,4 (m_X - m_H))."""
    return WINDOWS[band][2] / WINDOWS[NORM][2] * 10 ** (-0.4 * mag_band_minus_h)


# ----------------------------------------------------------------------------------------------
# 2. Empirische Hauptreihe [E]
# ----------------------------------------------------------------------------------------------
# SpT: (Klasse, Teff, V-Ic, V-Ks, J-H, H-Ks, Ks-W1, W1-W2 | None, M_Ks)
_MS = [
    ("F9V", "heisser_als_G", 6050, 0.620, 1.335, 0.236, 0.063, 0.057, -0.041, 2.915),
    ("G0V", "gelber_zwerg",  5930, 0.673, 1.437, 0.262, 0.067, 0.058, -0.043, 3.043),
    ("G2V", "gelber_zwerg",  5770, 0.713, 1.564, 0.293, 0.073, 0.061, -0.050, 3.236),
    ("G5V", "gelber_zwerg",  5660, 0.738, 1.635, 0.310, 0.076, 0.062, -0.052, 3.345),
    ("G8V", "gelber_zwerg",  5480, 0.786, 1.768, 0.342, 0.082, 0.065, -0.057, 3.532),
    ("G9V", "gelber_zwerg",  5380, 0.820, 1.857, 0.364, 0.087, 0.067, -0.060, 3.693),
    ("K0V", "k_zwerg",       5270, 0.853, 1.953, 0.387, 0.091, 0.070, -0.063, 3.827),
    ("K5V", "k_zwerg",       4440, 1.272, 2.880, 0.568, 0.132, 0.092, -0.073, 4.397),
    ("K7V", "k_zwerg",       4100, 1.519, 3.350, 0.619, 0.164, 0.110, None,   4.810),
    ("M0V", "roter_zwerg",   3850, 1.766, 3.650, 0.627, 0.190, 0.120, None,   5.150),
    ("M2V", "roter_zwerg",   3560, 2.173, 4.230, 0.600, 0.234, 0.140, None,   5.980),
    ("M4V", "roter_zwerg",   3210, 2.831, 5.250, 0.557, 0.280, 0.180, None,   7.360),
    ("M6V", "roter_zwerg",   2810, 4.100, 7.100, 0.605, 0.352, 0.225, 0.21,   9.220),
    ("M9V", "roter_zwerg",   2380, 4.660, 9.000, 0.720, 0.470, 0.330, 0.27,  10.400),
]


def _row_to_q(row) -> dict:
    _, _, _, v_ic, v_ks, j_h, h_ks, ks_w1, w1_w2, _ = row
    q = {
        "I":  color_to_ratio("I", (v_ks - v_ic) + h_ks),     # I-H = (I-Ks) + (H-Ks)
        "J":  color_to_ratio("J", j_h),
        "H":  1.0,
        "Ks": color_to_ratio("Ks", -h_ks),
        "W1": color_to_ratio("W1", -(h_ks + ks_w1)),
        "W2": color_to_ratio("W2", -(h_ks + ks_w1 + w1_w2)) if w1_w2 is not None else float("nan"),
    }
    return q


ANCHORS = [{"spt": r[0], "cls": r[1], "teff": r[2], "m_ks": r[9], "q": _row_to_q(r)} for r in _MS]

# Bereich der absoluten Ks-Helligkeit je Klasse [E]; nur fuer den Parallaxen-Test
M_KS_RANGE = {"gelber_zwerg": (3.043, 3.693), "roter_zwerg": (5.150, 10.400)}

# Rayleigh-Jeans [M]: q = (lambda_H / lambda)^2
RJ_Q = {b: (WINDOWS[NORM][0] / WINDOWS[b][0]) ** 2 for b in WINDOWS}

# Schnittgrenzen [V] (Herleitung: Mitte zwischen benachbarten Tabellenwerten)
W2_W1_METHANE = 1.4      # geometrische Mitte zwischen M9V (0,71) und T5V (2,66)

# Chi^2-Schwellen fuer p = 0,001 bei 1..5 Freiheitsgraden (Standardtabelle)
_CHI2_P001 = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52}


def _templates(steps: int = 10):
    """Hauptreihe fein unterteilt: log q linear zwischen benachbarten Spektraltypen interpoliert [V].
    Jede Zwischenstufe gehoert zur Klasse des naeheren Ankers -> Klassengrenze = Mitte."""
    out = []
    for a, b in zip(ANCHORS[:-1], ANCHORS[1:]):
        for k in range(steps):
            t = k / steps
            q = {band: math.exp((1 - t) * math.log(a["q"][band]) + t * math.log(b["q"][band]))
                 for band in STAR_BANDS}
            out.append({"cls": a["cls"] if t < 0.5 else b["cls"],
                        "near": a["spt"] if t < 0.5 else b["spt"],
                        "teff": (1 - t) * a["teff"] + t * b["teff"], "q": q})
    last = ANCHORS[-1]
    out.append({"cls": last["cls"], "near": last["spt"], "teff": last["teff"],
                "q": {band: last["q"][band] for band in STAR_BANDS}})
    return out


TEMPLATES = _templates()


# ----------------------------------------------------------------------------------------------
# 3. R = 200: Raster und Aufloesung (nur fuer Referenzprofile)
# ----------------------------------------------------------------------------------------------
def r_grid(r: float = 200.0, lam_min: float = 0.75, lam_max: float = 5.0) -> np.ndarray:
    """Formel 1: logarithmisches Raster mit zwei Stuetzstellen je Aufloesungselement."""
    n = int(math.ceil(math.log(lam_max / lam_min) / math.log(1 + 1 / (2 * r))))
    return lam_min * (1 + 1 / (2 * r)) ** np.arange(n)


def smooth_to_r(wave, flux, r_native: float, r_target: float = 200.0) -> np.ndarray:
    """Formel 2: senkt die Aufloesung eines Spektrums von r_native auf r_target (Gauss in ln lambda).
    Eine Erhoehung ist physikalisch unmoeglich und loest einen Fehler aus."""
    if r_native <= r_target:
        raise ValueError(f"Aufloesung laesst sich nicht erhoehen: R_nativ = {r_native} <= R_ziel = {r_target}.")
    wave, flux = np.asarray(wave, float), np.asarray(flux, float)
    sigma = math.sqrt(1 / r_target ** 2 - 1 / r_native ** 2) / 2.3548
    ln = np.log(wave)
    out = np.empty_like(flux)
    for i, x in enumerate(ln):
        w = np.exp(-0.5 * ((ln - x) / sigma) ** 2)
        w[~np.isfinite(flux)] = 0.0
        out[i] = np.nansum(w * flux) / w.sum() if w.sum() > 0 else np.nan
    return out


def reference_profile(spt: str, r: float = 200.0, scale: float = 1.0):
    """Referenzprofil f(lambda) = scale * F_nu / <F_nu>_H auf dem R-Raster.
    Zwischen den sechs empirischen Stuetzfenstern wird in log-log interpoliert [V]; das Profil
    enthaelt daher KEINE Molekuelbanden. Ausserhalb 0,80-4,60 um wird nicht extrapoliert (NaN)."""
    if spt == "RJ":
        grid = r_grid(r)
        return grid, scale * (WINDOWS[NORM][0] / grid) ** 2
    anchor = next(a for a in ANCHORS if a["spt"] == spt)
    pts = [(WINDOWS[b][0], anchor["q"][b]) for b in WINDOWS if np.isfinite(anchor["q"][b])]
    lam, q = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    grid = r_grid(r)
    prof = np.exp(np.interp(np.log(grid), np.log(lam), np.log(q), left=np.nan, right=np.nan))
    return grid, scale * prof


# ----------------------------------------------------------------------------------------------
# 4. Messung -> Fensterfluesse -> Verhaeltnisse
# ----------------------------------------------------------------------------------------------
def window_fluxes(wave_um, flux_mjy, err_mjy) -> dict:
    """Je Stuetzfenster: Fluss an der Referenzwellenlaenge und sein Fehler.
    Ab 3 Messpunkten: gewichtete Gerade F(ln lambda), ausgewertet bei der Referenzwellenlaenge
    (gleicht ungleichmaessige Abtastung aus). Sonst gewichtetes Mittel."""
    w, f, e = (np.asarray(a, float) for a in (wave_um, flux_mjy, err_mjy))
    ok = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    out = {}
    for band, (ref, (lo, hi), _) in WINDOWS.items():
        m = ok & (w >= lo) & (w <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        wt = 1 / e[m] ** 2
        x = np.log(w[m]) - math.log(ref)
        sw = wt.sum()
        xbar = (wt * x).sum() / sw
        sxx = (wt * (x - xbar) ** 2).sum()
        mean = (wt * f[m]).sum() / sw
        if n >= 3 and sxx > 0 and (x.max() - x.min()) > 0.3 * math.log(hi / lo):
            slope = (wt * (x - xbar) * f[m]).sum() / sxx
            val = mean - slope * xbar                       # Wert bei x = 0, also bei ref
            sig = math.sqrt(1 / sw + xbar ** 2 / sxx)
            method = "gerade"
        else:
            val, sig, method = mean, math.sqrt(1 / sw), "mittel"
        out[band] = {"flux": float(val), "err": float(sig), "n": n, "method": method}
    return out


def ratios(win: dict, flux_floor: float = 0.05) -> dict:
    """Formel 3: q_X = F_X / F_H mit Fehler. flux_floor = systematische Untergrenze je Verhaeltnis
    (5 % wie FLUX_FLOOR in der Weboberflaeche) [V]."""
    h = win[NORM]
    q = {}
    for band, v in win.items():
        if band == NORM or v["flux"] <= 0:
            continue
        val = v["flux"] / h["flux"]
        rel = math.sqrt((v["err"] / v["flux"]) ** 2 + (h["err"] / h["flux"]) ** 2 + flux_floor ** 2)
        q[band] = (val, rel)                                # rel = sigma(ln q)
    return q


# ----------------------------------------------------------------------------------------------
# 5. Klassifikation
# ----------------------------------------------------------------------------------------------
def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def classify(wave_um, flux_mjy, err_mjy, parallax_mas: float | None = None,
             flux_floor: float = 0.05,          # [V] systematische Untergrenze je Flussverhaeltnis
             min_snr_h: float = 10.0,           # [V] Mindest-S/N im Normierungsfenster H
             min_points_per_window: int = 2,    # [V]
             p_clear: float = 0.95,             # [V] ab hier "eindeutig"
             p_likely: float = 0.68,            # [V] ab hier "wahrscheinlich"
             m_ks_tolerance: float = 1.0):      # [V] erlaubte Abweichung von M_Ks (mag)
    """
    Rueckgabe: dict mit
      label        : 'gelber_zwerg' | 'roter_zwerg' | 'methan_atmosphaere' | 'k_zwerg' |
                     'heisser_als_G' | 'nicht_zugeordnet' | 'nicht_klassifizierbar'
      status       : 'eindeutig' | 'wahrscheinlich' | 'unsicher' | 'abgelehnt'
      probability  : Wahrscheinlichkeit des Labels unter den geprueften Hypothesen
      probabilities: je Sternklasse
      nearest_type, teff_k, chi2, dof, q, warnings
    Die Wahrscheinlichkeiten setzen gleiche Vorab-Haeufigkeit aller Hauptreihen-Vorlagen voraus [V].
    """
    res = {"label": "nicht_klassifizierbar", "status": "abgelehnt", "probability": None,
           "probabilities": {}, "nearest_type": None, "teff_k": None, "chi2": None, "dof": None,
           "q": {}, "warnings": []}
    warn = res["warnings"]
    win = window_fluxes(wave_um, flux_mjy, err_mjy)

    # --- Stufe 0: Messqualitaet -------------------------------------------------------------
    if NORM not in win or win[NORM]["n"] < min_points_per_window:
        warn.append("Too few data points in the H window (1.49-1.78 um); normalisation not possible.")
        return res
    snr_h = win[NORM]["flux"] / win[NORM]["err"]
    if snr_h < min_snr_h:
        warn.append(f"S/N in the H window only {snr_h:.1f} (required {min_snr_h:g}).")
        return res
    thin = [b for b, v in win.items() if v["n"] < min_points_per_window]
    for b in thin:
        del win[b]
    if thin:
        warn.append("Windows with too few data points ignored: " + ", ".join(thin))
    q = ratios(win, flux_floor)
    res["q"] = {b: {"value": round(v, 4), "rel_err": round(s, 4)} for b, (v, s) in q.items()}

    # --- Stufe 1: kuehle Methan-Atmosphaere (Gasplanet-Analogon) ---------------------------------
    if "W1" in win and "W2" in win and win["W2"]["flux"] > 0:
        f1, f2 = win["W1"], win["W2"]
        if f1["flux"] <= 0:
            p_m = 1.0 if f2["flux"] / f2["err"] >= 5 else 0.0
        else:
            ln_r = math.log(f2["flux"] / f1["flux"])
            s_r = math.sqrt((f1["err"] / f1["flux"]) ** 2 + (f2["err"] / f2["flux"]) ** 2 + flux_floor ** 2)
            p_m = _norm_cdf((ln_r - math.log(W2_W1_METHANE)) / s_r)
        if p_m >= p_likely:
            res.update(label="methan_atmosphaere", probability=round(p_m, 4),
                       status="eindeutig" if p_m >= p_clear else "wahrscheinlich")
            warn.append("Spectral type like a T/Y dwarf. Whether planet or brown dwarf cannot be decided "
                        "from the spectrum; dust-enshrouded stars and active galaxies can be similarly red.")
            return res
    else:
        warn.append("W1 or W2 missing: methane atmosphere test not possible.")

    # --- Stufe 2: Hauptreihen-Fit --------------------------------------------------------------
    bands = [b for b in STAR_BANDS if b in q]
    if "I" not in bands or len(bands) < 3:
        warn.append("The star classes need the I window (0.75-0.88 um) and at least two more "
                    "windows from J, Ks, W1.")
        return res
    chi2 = np.array([sum(((math.log(q[b][0]) - math.log(t["q"][b])) / q[b][1]) ** 2 for b in bands)
                     for t in TEMPLATES])
    best = int(np.argmin(chi2))
    like = np.exp(-0.5 * (chi2 - chi2[best]))
    classes = sorted({t["cls"] for t in TEMPLATES})
    lmax = {c: max(like[i] for i, t in enumerate(TEMPLATES) if t["cls"] == c) for c in classes}
    total = sum(lmax.values())
    probs = {c: lmax[c] / total for c in classes}
    dof = len(bands) - 1
    tb = TEMPLATES[best]
    res.update(probabilities={c: round(p, 4) for c, p in probs.items()}, nearest_type=tb["near"],
               teff_k=round(tb["teff"]), chi2=round(float(chi2[best]), 2), dof=dof)

    # schlechter Fit -> keine der Vorlagen passt
    if chi2[best] > _CHI2_P001[dof]:
        res.update(label="nicht_zugeordnet", status="abgelehnt", probability=None)
        hot = all(b in q for b in ("I", "J")) and q["I"][0] > ANCHORS[0]["q"]["I"]
        warn.append("No main-sequence profile fits (p < 0.001)."
                    + (" Very blue source: hot star or white dwarf. A neutron star would have the same "
                       "Rayleigh-Jeans profile but would be orders of magnitude too faint for SPHEREx." if hot else
                       " Possible causes: giant, reddening, dust, galaxy, blend of two sources, variability."))
        return res

    label = max(probs, key=probs.get)
    p = probs[label]
    res.update(label=label, probability=round(p, 4),
               status="eindeutig" if p >= p_clear else "wahrscheinlich" if p >= p_likely else "unsicher")
    if best in (0, len(TEMPLATES) - 1):
        warn.append("Best profile is at the edge of the template grid (hotter than F9 or cooler than M9).")

    # --- Stufe 3: Leuchtkraftklasse mit Parallaxe ------------------------------------------------
    if label in M_KS_RANGE:
        if parallax_mas is None or parallax_mas <= 0 or "Ks" not in win:
            warn.append("Without parallax 'dwarf' is not verified: a giant of the same colour is not ruled out.")
        else:
            ks_mag = -2.5 * math.log10(win["Ks"]["flux"] * 1e-3 / WINDOWS["Ks"][2])
            m_ks = ks_mag + 5 * math.log10(parallax_mas) - 10
            lo, hi = M_KS_RANGE[label]
            res["m_ks"] = round(m_ks, 2)
            if not (lo - m_ks_tolerance <= m_ks <= hi + m_ks_tolerance):
                res["status"] = "unsicher"
                warn.append(f"M_Ks = {m_ks:.2f} mag is outside {lo:.1f}-{hi:.1f} mag for this class: "
                            "probably not a main-sequence star (giant, subgiant or binary).")
    return res


def classify_frames(frames: list[dict], **kwargs):
    """Bequemer Einstieg: Liste der JSON-Antworten von /api/load_spherex_mef.
    Filtert wie die Weboberflaeche (Quelle gefunden, nicht ueberbelichtet, Zielpixel in Ordnung)."""
    rows = [(f["exact_wavelength_um"], f["aperture_flux_mjy"], f["aperture_flux_err_mjy"]) for f in frames
            if f.get("source_detected") and not f.get("saturated") and not f.get("target_pixel_bad")
            and f.get("aperture_flux_mjy") is not None and (f.get("aperture_flux_err_mjy") or 0) > 0]
    if not rows:
        return {"label": "nicht_klassifizierbar", "status": "abgelehnt", "warnings": ["No usable measurement."]}
    w, fl, e = zip(*rows)
    return classify(w, fl, e, **kwargs)


# Einbau in spherex_api.py (Vorschlag):
#
#   from spherex_classify import classify
#   class ClassifyRequest(BaseModel):
#       wave_um: list[float]; flux_mjy: list[float]; err_mjy: list[float]; parallax_mas: float | None = None
#   @app.post("/api/classify")
#   def classify_spectrum(req: ClassifyRequest):
#       return classify(req.wave_um, req.flux_mjy, req.err_mjy, req.parallax_mas)
