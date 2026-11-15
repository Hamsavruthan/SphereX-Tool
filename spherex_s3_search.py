"""
SPHEREx-Suche direkt im AWS-Bucket "nasa-irsa-spherex" - ohne IRSA.

Wozu: Die IRSA-Suchdienste (SIA, TAP) koennen ausfallen. Im AWS-Bucket gibt es fuer SPHEREx aber keinen
Positionsindex. Dieses Modul baut sich deshalb selbst einen, nur fuer die Himmelsgegend, die gesucht wird,
und speichert ihn in %LOCALAPPDATA%/SphereX/s3_index.sqlite (bewusst ausserhalb von OneDrive). Die zweite Suche in derselben Gegend
ist dann schnell.

Ablauf einer Suche:
  1. Wochenordner auflisten (dr1/level2/2025W31_2A, ...). Der Ordnername enthaelt die Kalenderwoche.
  2. Nur Wochen behalten, in denen SPHEREx das Ziel ueberhaupt sehen kann. SPHEREx beobachtet ungefaehr
     senkrecht zur Sonnenrichtung (Elongation um 90 Grad). Aus der Wochenmitte folgt die Sonnenposition,
     daraus die Elongation des Ziels.
  3. Grobsuche: Nur den Ordner von Detektor 2 auflisten. Je "grossem Schwenk" (Teil 0156 im Namen
     level2_2025W31_2A_0156_1D2_...) den Dateikopf EINER Aufnahme lesen (nur die ersten 32 KB der Datei)
     und den Abstand der Bildmitte zum Ziel ausrechnen.
  4. Feinsuche: Nur fuer Wochen mit Schwenks naeher als 8 Grad die uebrigen Detektorordner auflisten,
     dann die Koepfe ALLER Aufnahmen dieser Schwenks lesen (alle kleinen Schwenks, alle 6 Detektoren),
     die vier Bildecken mit dem vollen WCS berechnen und pruefen, ob das Ziel im Bild liegt.
  5. Ergebnis als Tabelle mit denselben Spalten wie die IRSA-Suche.

Test von der Kommandozeile:
    python spherex_s3_search.py 269.452 4.6933            (Barnard's Star, Grad)
    python spherex_s3_search.py 269.452 4.6933 --releases dr1

Pakete: s3fs, astropy, numpy, pandas
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

BUCKET = "nasa-irsa-spherex"
REGION = "us-east-1"
# Index NICHT im OneDrive-/Projektordner: OneDrive sperrt Dateien beim Synchronisieren, dann haengt SQLite.
_INDEX_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "SphereX")
INDEX_PATH = os.environ.get("SPHEREX_S3_INDEX", os.path.join(_INDEX_DIR, "s3_index.sqlite"))

COARSE_BYTES = 32 * 1024          # Grobsuche: Anfang des Dateikopfs reicht fuer CRVAL und Zeiten
FULL_BYTES = 96 * 1024            # Feinsuche: ganzer Kopf der IMAGE-Ebene (mit SIP-Verzeichnung)
COARSE_RADIUS_DEG = 8.0           # Streifen 3,5 x 11,3 Grad: Ecke bis ~5,9 Grad von der Mitte, plus Reserve
ELONG_STEPS_DEG = (20.0, 45.0)    # erst enges Elongationsfenster, bei null Treffern weiter
MAX_WEEK_DIRS = int(os.environ.get("SPHEREX_S3_MAX_DIRS", "160"))
LISTING_MAX_AGE_DAYS = 1.0        # Liste der Wochenordner taeglich auffrischen (neue Daten)
WEEK_LISTING_MAX_AGE_DAYS = 30.0  # Inhalt eines Wochenordners aendert sich kaum

# Wellenlaengenbereiche der Detektoren in Metern (Crill et al. 2024, Tab. 2)
DETECTOR_RANGE_M = {1: (0.75e-6, 1.12e-6), 2: (1.10e-6, 1.65e-6), 3: (1.63e-6, 2.44e-6),
                    4: (2.40e-6, 3.85e-6), 5: (3.81e-6, 4.43e-6), 6: (4.41e-6, 5.01e-6)}

WEEK_RE = re.compile(r"^(\d{4})W(\d{2})_(\w+)$")
VERSION_RE = re.compile(r"l2b-v(\d+)-(\d{4})-(\d{3})")
FILE_RE = re.compile(r"level2_(\d{4}W\d{2}_\w+?)_(\d{4})_(\d+)D(\d)_spx_.+\.fits$")
CARD_RE = re.compile(r"^([A-Z0-9_-]{1,8})\s*=\s*('(?:[^']|'')*'|[^/]*)")

_db_lock = threading.Lock()
_fs = None


def log(msg: str):
    print(f"[AWS-Suche] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Zugang zu S3 und zum lokalen Index
# ----------------------------------------------------------------------------------------------
def fs():
    global _fs
    if _fs is None:
        import s3fs
        _fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": REGION},
                                config_kwargs={"connect_timeout": 20, "read_timeout": 60,
                                               "retries": {"max_attempts": 5}})
    return _fs


def _db():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    con = sqlite3.connect(INDEX_PATH, timeout=60, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS listing (prefix TEXT PRIMARY KEY, saved REAL, entries TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS header (key TEXT PRIMARY KEY, det INTEGER, obsid TEXT, "
                "mjd_beg REAL, mjd_end REAL, xposure REAL, ra REAL, dec REAL, corners TEXT)")
    return con


def _strip_bucket(path: str) -> str:
    return path[len(BUCKET) + 1:] if path.startswith(BUCKET + "/") else path


def _list(prefix: str, max_age_days: float, recursive: bool = False) -> list[str]:
    """S3-Schluessel unter `prefix` (mit Zwischenspeicher in der Datenbank)."""
    con = _db()
    row = con.execute("SELECT saved, entries FROM listing WHERE prefix = ?", (prefix,)).fetchone()
    if row and time.time() - row[0] < max_age_days * 86400:
        con.close()
        return json.loads(row[1])
    path = f"{BUCKET}/{prefix}"
    entries = fs().find(path) if recursive else fs().ls(path, detail=False, refresh=True)
    entries = [_strip_bucket(e) for e in entries]
    with _db_lock:
        con.execute("INSERT OR REPLACE INTO listing VALUES (?, ?, ?)", (prefix, time.time(), json.dumps(entries)))
        con.commit()
    con.close()
    return entries


# ----------------------------------------------------------------------------------------------
# Himmelsmechanik (Naeherungen reichen fuer die Vorauswahl der Wochen)
# ----------------------------------------------------------------------------------------------
OBLIQUITY = math.radians(23.4393)


def ecliptic(ra_deg: float, dec_deg: float):
    """Aequatoriale Koordinaten -> ekliptikale Laenge und Breite (Grad)."""
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    sin_b = math.sin(d) * math.cos(OBLIQUITY) - math.cos(d) * math.sin(OBLIQUITY) * math.sin(a)
    lam = math.atan2(math.sin(a) * math.cos(OBLIQUITY) + math.tan(d) * math.sin(OBLIQUITY), math.cos(a))
    return math.degrees(lam) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))


def sun_longitude(mjd: float) -> float:
    """Ekliptikale Laenge der Sonne (Astronomical Almanac, Genauigkeit ~0,01 Grad)."""
    n = mjd - 51544.5
    L = 280.460 + 0.9856474 * n
    g = math.radians(357.528 + 0.9856003 * n)
    return (L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0


def elongation(lam: float, beta: float, mjd: float) -> float:
    """Winkel Sonne-Ziel (Grad): cos e = cos(beta) * cos(lambda - lambda_Sonne)."""
    c = math.cos(math.radians(beta)) * math.cos(math.radians(lam - sun_longitude(mjd)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def week_mid_mjd(year: int, week: int) -> float:
    monday = dt.date.fromisocalendar(year, week, 1)
    return (monday - dt.date(1858, 11, 17)).days + 3.5


def separation(ra1, dec1, ra2, dec2):
    """Winkelabstand in Grad (Haversine), auch fuer numpy-Felder."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    h = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def inside_footprint(ra, dec, corners) -> bool:
    """Liegt (ra, dec) im Viereck der vier Bildecken? Gnomonische Projektion um die Viereckmitte."""
    c = np.array(corners, dtype=float)
    vec = np.stack([np.cos(np.radians(c[:, 1])) * np.cos(np.radians(c[:, 0])),
                    np.cos(np.radians(c[:, 1])) * np.sin(np.radians(c[:, 0])),
                    np.sin(np.radians(c[:, 1]))], axis=1).mean(axis=0)
    ra0 = math.degrees(math.atan2(vec[1], vec[0]))
    dec0 = math.degrees(math.atan2(vec[2], math.hypot(vec[0], vec[1])))

    def proj(r, d):
        r, d, r0, d0 = map(math.radians, (r, d, ra0, dec0))
        cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(r - r0)
        if cos_c <= 0:
            return None
        return (math.cos(d) * math.sin(r - r0) / cos_c,
                (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(r - r0)) / cos_c)

    p = proj(ra, dec)
    poly = [proj(r, d) for r, d in c]
    if p is None or any(q is None for q in poly):
        return False
    signs = []
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        signs.append((x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


# ----------------------------------------------------------------------------------------------
# Dateikoepfe lesen
# ----------------------------------------------------------------------------------------------
def _split_headers(raw: bytes):
    """Erste Bytes einer SPHEREx-Datei -> (Text des IMAGE-Kopfs, vollstaendig?)."""
    text = raw.decode("ascii", errors="replace")

    def find_end(start):
        for i in range(start, len(text) - 79, 80):
            if text[i:i + 3] == "END" and text[i:i + 80].strip() == "END":
                return i
        return None

    e1 = find_end(0)                      # Ende des (leeren) Hauptkopfs
    if e1 is None:
        return "", False
    start = (e1 // 2880 + 1) * 2880       # Koepfe sind in 2880-Byte-Bloecken abgelegt
    e2 = find_end(start)
    if e2 is None:
        usable = (len(text) - start) // 80 * 80
        return text[start:start + usable], False
    return text[start:e2 + 80], True


def _cards(text: str) -> dict:
    out = {}
    for i in range(0, len(text) - 79, 80):
        m = CARD_RE.match(text[i:i + 80])
        if m:
            value = m.group(2).strip()
            if value.startswith("'"):
                value = value.strip("'").replace("''", "'").strip()
            out[m.group(1)] = value
    return out


def _num(cards, key):
    try:
        return float(cards[key])
    except (KeyError, ValueError):
        return None


def _footprint(text: str):
    """Vollstaendiger IMAGE-Kopf -> vier Bildecken (RA, Dec) mit SIP-Verzeichnung."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = fits.Header.fromstring(text)
        w = WCS(header)
        nx, ny = int(header.get("NAXIS1", 2040)), int(header.get("NAXIS2", 2040))
        xs = np.array([0, nx - 1, nx - 1, 0], dtype=float)
        ys = np.array([0, 0, ny - 1, ny - 1], dtype=float)
        ra, dec = w.all_pix2world(xs, ys, 0)
    return [[round(float(r), 6), round(float(d), 6)] for r, d in zip(ra, dec)]


def read_headers(keys: list[str], full: bool) -> dict:
    """
    Liest die Koepfe der Dateien `keys` (nur Byte-Bereiche, viele gleichzeitig) und schreibt sie in den Index.
    full=False: Mitte und Zeiten.  full=True: zusaetzlich die vier Bildecken.
    Rueckgabe: {key: (det, obsid, mjd_beg, mjd_end, xposure, ra, dec, corners_json)}
    """
    con = _db()
    known = {}
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = f"SELECT * FROM header WHERE key IN ({','.join('?' * len(part))})"
        for row in con.execute(q, part):
            if not full or row[8]:
                known[row[0]] = row[1:]
    todo = [k for k in keys if k not in known]
    if todo:
        log(f"lese {len(todo)} Dateikoepfe ({'voll' if full else 'grob'}), {len(known)} schon im Index ...")
    nbytes = FULL_BYTES if full else COARSE_BYTES
    t0 = time.time()
    for i in range(0, len(todo), 256):
        part = todo[i:i + 256]
        paths = [f"{BUCKET}/{k}" for k in part]
        results = fs().cat_ranges(paths, [0] * len(paths), [nbytes] * len(paths), on_error="return")
        rows = []
        for key, raw in zip(part, results):
            if isinstance(raw, BaseException):
                log(f"  uebersprungen ({type(raw).__name__}): {key}")
                continue
            text, complete = _split_headers(raw)
            if full and not complete:
                try:
                    raw = fs().cat_file(f"{BUCKET}/{key}", start=0, end=4 * nbytes)
                    text, complete = _split_headers(raw)
                except Exception as e:
                    log(f"  Kopf unvollstaendig ({e!r}): {key}")
            c = _cards(text)
            ra, dec = _num(c, "CRVAL1"), _num(c, "CRVAL2")
            if ra is None or dec is None:
                continue
            corners = None
            if full and complete:
                try:
                    corners = json.dumps(_footprint(text))
                except Exception as e:
                    log(f"  WCS nicht lesbar ({e!r}): {key}")
            det = int(_num(c, "DETECTOR") or 0)
            row = (key, det, c.get("OBSID", ""), _num(c, "MJD-BEG") or _num(c, "MJD-OBS"),
                   _num(c, "MJD-END"), _num(c, "XPOSURE"), ra, dec, corners)
            rows.append(row)
            known[key] = row[1:]
        with _db_lock:
            con.executemany("INSERT OR REPLACE INTO header VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            con.commit()
        done = min(i + 256, len(todo))
        log(f"  {done}/{len(todo)} Koepfe ({time.time() - t0:.0f} s)")
    con.close()
    return known


# ----------------------------------------------------------------------------------------------
# Suche
# ----------------------------------------------------------------------------------------------
def _week_dirs(release: str):
    """[(Schluessel des Wochenordners, Wochenmitte als MJD)] fuer ein Release."""
    out = []
    for entry in _list(f"{release}/level2", LISTING_MAX_AGE_DAYS):
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        m = WEEK_RE.match(name)
        if m:
            try:
                out.append((entry.rstrip("/"), week_mid_mjd(int(m.group(1)), int(m.group(2)))))
            except ValueError:
                pass
    return out


def _version_dir(week_key: str):
    """Neueste Verarbeitungsversion in einem Wochenordner, z. B. .../2025W31_2A/l2b-v20-2025-264."""
    versions = [e.rstrip("/") for e in _list(week_key, WEEK_LISTING_MAX_AGE_DAYS) if VERSION_RE.search(e)]
    if not versions:
        return None
    return max(versions, key=lambda v: tuple(int(x) for x in VERSION_RE.search(v).groups()))


def _files(week_key: str, det: int) -> list[str]:
    """FITS-Dateien EINES Detektors in einem Wochenordner (Unterordner .../<Version>/<Detektor>/)."""
    version = _version_dir(week_key)
    if not version:
        return []
    try:
        return [k for k in _list(f"{version}/{det}", WEEK_LISTING_MAX_AGE_DAYS) if k.endswith(".fits")]
    except FileNotFoundError:
        return []


def _list_many(week_keys: list[str], dets: tuple, label: str) -> dict:
    """Listet die Detektorordner vieler Wochen parallel und meldet den Fortschritt."""
    jobs = [(w, d) for w in week_keys for d in dets]
    out, t0 = {}, time.time()
    log(f"liste {len(jobs)} Ordner auf ({label}) ...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_files, w, d): (w, d) for w, d in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            w, d = futures[fut]
            try:
                out[(w, d)] = fut.result()
            except Exception as e:
                log(f"  Ordner nicht lesbar ({e!r}): {w} / {d}")
                out[(w, d)] = []
            if n % 5 == 0 or n == len(jobs):
                log(f"  {n}/{len(jobs)} Ordner ({sum(len(v) for v in out.values())} Dateien, {time.time() - t0:.0f} s)")
    return out


def find_images(ra: float, dec: float, releases: list[str]) -> list[dict]:
    """Alle Spektralbilder in `releases` (z. B. ['dr1', 'qr3']), deren Bildflaeche (ra, dec) enthaelt."""
    lam, beta = ecliptic(ra, dec)
    log(f"Ziel RA {ra:.5f}, Dec {dec:.5f} = ekliptikal {lam:.2f} / {beta:+.2f} Grad")
    log(f"Index: {INDEX_PATH}")
    all_weeks = []
    for rel in releases:
        try:
            all_weeks += [(rel, k, m) for k, m in _week_dirs(rel)]
        except FileNotFoundError:
            log(f"Release '{rel}' hat im Bucket keinen Ordner level2 - uebersprungen.")
    log(f"{len(all_weeks)} Wochenordner in {', '.join(releases)}")
    searched = set()

    for tol in ELONG_STEPS_DEG:
        weeks = []
        for rel, key, mjd in all_weeks:
            dev = min(abs(elongation(lam, beta, mjd + d) - 90.0) for d in (-3.5, 0.0, 3.5))
            if dev <= tol and key not in searched:
                weeks.append((dev, rel, key, mjd))
        if len(weeks) > MAX_WEEK_DIRS:
            log(f"{len(weeks)} Wochenordner kommen in Frage (Ziel nahe am Ekliptikpol?) - "
                f"durchsuche nur die {MAX_WEEK_DIRS} neuesten (SPHEREX_S3_MAX_DIRS).")
            weeks = sorted(weeks, key=lambda w: -w[3])[:MAX_WEEK_DIRS]
        log(f"Elongationsfenster 90 +- {tol:.0f} Grad: {len(weeks)} Wochenordner")
        if not weeks:
            continue
        searched.update(w[2] for w in weeks)
        release_of_week = {w[2]: w[1] for w in weeks}

        # Grobsuche nur mit Detektor 2 (Mitte des kurzwelligen Streifens)
        listing = _list_many([w[2] for w in weeks], (2,), "Detektor 2")
        groups: dict[tuple, list] = {}
        week_of_group = {}
        for (week_key, _), files in listing.items():
            for f in files:
                m = FILE_RE.search(f)
                if m:
                    g = (m.group(1), m.group(2))
                    groups.setdefault(g, []).append((int(m.group(3)), f))
                    week_of_group[g] = week_key
        log(f"{len(groups)} grosse Schwenks")
        reps = {g: sorted(items)[0][1] for g, items in groups.items()}
        coarse = read_headers(list(reps.values()), full=False)
        near = [g for g, k in reps.items() if k in coarse
                and separation(ra, dec, coarse[k][5], coarse[k][6]) < COARSE_RADIUS_DEG]
        log(f"Grobsuche: {len(near)} Schwenks naeher als {COARSE_RADIUS_DEG:.0f} Grad")
        if not near:
            continue

        # Feinsuche: alle sechs Detektoren, aber nur in den Wochen mit nahen Schwenks
        near_set = set(near)
        near_weeks = sorted({week_of_group[g] for g in near})
        more = _list_many(near_weeks, (1, 3, 4, 5, 6), "uebrige Detektoren")
        more.update({k: v for k, v in listing.items() if k[0] in near_weeks})
        fine_keys, release_of = [], {}
        for (week_key, _), files in more.items():
            for f in files:
                m = FILE_RE.search(f)
                if m and (m.group(1), m.group(2)) in near_set:
                    fine_keys.append(f)
                    release_of[f] = release_of_week[week_key]
        fine = read_headers(fine_keys, full=True)
        hits = []
        for key in fine_keys:
            row = fine.get(key)
            if not row or not row[7]:
                continue
            det, obsid, mjd_beg, mjd_end, xposure, cra, cdec, corners = row
            if inside_footprint(ra, dec, json.loads(corners)):
                hits.append({"key": key, "release": release_of[key], "det": det, "obsid": obsid,
                             "mjd_beg": mjd_beg, "mjd_end": mjd_end, "xposure": xposure})
        log(f"Feinsuche: {len(hits)} Aufnahmen enthalten das Ziel")
        if hits:
            return hits
    return []


def search_dataframe(ra: float, dec: float, collections: list[str]):
    """Ergebnis im Format der IRSA-SIA-Suche (Spalten wie in spherex_api.search_spherex erwartet)."""
    import pandas as pd
    releases = []
    for c in collections:
        rel = c.strip().removeprefix("spherex_")
        if rel.endswith("_deep"):
            log(f"'{c}' (Deep Survey) wird von der AWS-Suche nicht unterstuetzt - uebersprungen.")
            continue
        if rel and rel not in releases:
            releases.append(rel)
    hits = find_images(ra, dec, releases)
    rows = []
    for h in hits:
        m = FILE_RE.search(h["key"])
        obs_id = f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else h["obsid"].replace(" ", "_")
        lo, hi = DETECTOR_RANGE_M.get(h["det"], (np.nan, np.nan))
        rows.append({
            "access_url": f"https://irsa.ipac.caltech.edu/ibe/data/spherex/{h['key']}",
            "cloud_access": json.dumps({"aws": {"bucket_name": BUCKET, "key": h["key"], "region": REGION}}),
            "t_min": h["mjd_beg"],
            "t_max": h["mjd_end"] if h["mjd_end"] is not None else h["mjd_beg"],
            "t_exptime": h["xposure"] or 0.0,
            "energy_bandpassname": f"SPHEREx-D{h['det']}",
            "em_min": lo, "em_max": hi,
            "obs_id": obs_id,
            "obs_collection": f"spherex_{h['release']}",
            "dataproduct_type": "image",
        })
    return pd.DataFrame(rows, columns=["access_url", "cloud_access", "t_min", "t_max", "t_exptime",
                                       "energy_bandpassname", "em_min", "em_max", "obs_id",
                                       "obs_collection", "dataproduct_type"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    rels = ["dr1", "qr3"]
    if "--releases" in sys.argv:
        rels = sys.argv[sys.argv.index("--releases") + 1].split(",")
    t = time.time()
    df = search_dataframe(float(sys.argv[1]), float(sys.argv[2]), [f"spherex_{r}" for r in rels])
    print(f"\n{len(df)} Aufnahmen in {time.time() - t:.0f} s")
    if len(df):
        df["det"] = df["energy_bandpassname"].str[-1]
        print(df.groupby("det").size().rename("Aufnahmen je Detektor").to_string())
        print("Zeitraum MJD", round(df["t_min"].min(), 2), "bis", round(df["t_min"].max(), 2))
        print(df[["obs_id", "energy_bandpassname", "t_min"]].head(10).to_string(index=False))