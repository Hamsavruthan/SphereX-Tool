"""
SPHEREx Grundsystem API  (v2.2 - Bilder aus dem AWS-Bucket von IRSA)

Start:   python spherex_api.py          ->  http://127.0.0.1:8000/docs
Pakete:  pip install fastapi uvicorn astropy astroquery numpy pillow pandas s3fs

Neu in v2.2:
  * Die Spektralbilder werden direkt aus dem oeffentlichen AWS-Bucket "nasa-irsa-spherex"
    (Region us-east-1, ohne Konto/Schluessel) gelesen statt ueber den IRSA-Ausschnittdienst.
    Gelesen wird nur der Bildausschnitt um das Ziel (Byte-Bereiche), nicht die ganze Datei.
  * Umschalten mit der Umgebungsvariable SPHEREX_IMAGE_SOURCE:
        aws  = nur AWS
        auto = erst AWS, bei Fehler IRSA-Ausschnittdienst (Voreinstellung)
        irsa = nur IRSA (Verhalten wie v2.1)
  * SUCHE (welche Bilder decken das Ziel ab?), Umgebungsvariable SPHEREX_SEARCH:
        auto = IRSA (SIA, dann TAP); faellt IRSA aus: gespeichertes Ergebnis, sonst eigene
               Suche im AWS-Bucket (spherex_s3_search.py). Voreinstellung.
        aws  = immer die eigene AWS-Suche, IRSA wird nicht gefragt.
        irsa = nur IRSA.
    Nach einem IRSA-Ausfall wird IRSA 10 Minuten lang nicht mehr gefragt (spart je Suche ~20 s).
"""

# --- 1. IMPORTE ---
import io
import math
import hashlib
import os
import re
import time
import base64
import json
import http.client
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.utils.data import conf
from astroquery.ipac.irsa import Irsa

from spherex_classify import classify as classify_spectrum   # liegt im selben Ordner
import spherex_s3_search                                      # liegt im selben Ordner

# IRSA-Tutorial: Lesezugriffe koennen das astropy-Standardlimit (10 s) ueberschreiten
conf.remote_timeout = 120


# --- 2. KONFIGURATION ---
COLLECTIONS = os.environ.get("SPHEREX_COLLECTIONS", "spherex_dr1,spherex_qr3").split(",")

# Kataloge (SIMBAD / Gaia-Reserve) standardmaessig aktiv ("1"), damit Namenssuchen wie
# "Barnard's Star" automatisch von Epoche J2000 auf das Beobachtungsjahr verschoben werden.
# Abschalten mit Umgebungsvariable SPHEREX_USE_CATALOGS=0.
USE_CATALOGS = os.environ.get("SPHEREX_USE_CATALOGS", "1") == "1"

# Woher die Bilder kommen (siehe Kopf der Datei)
IMAGE_SOURCE = os.environ.get("SPHEREX_IMAGE_SOURCE", "auto").strip().lower()
if IMAGE_SOURCE not in ("aws", "auto", "irsa"):
    IMAGE_SOURCE = "auto"
S3_BUCKET = "nasa-irsa-spherex"
S3_REGION = "us-east-1"
S3_FSSPEC_KWARGS = {
    "anon": True,                                   # oeffentlicher Bucket: ohne AWS-Konto lesen
    "client_kwargs": {"region_name": S3_REGION},
    "default_block_size": 1 << 20,                  # 1 MB je Lesezugriff reicht fuer einen Ausschnitt
    "config_kwargs": {"connect_timeout": 20, "read_timeout": 120, "retries": {"max_attempts": 3}},
}
SEARCH_MODE = os.environ.get("SPHEREX_SEARCH", "auto").strip().lower()
if SEARCH_MODE not in ("auto", "aws", "irsa"):
    SEARCH_MODE = "auto"
IRSA_PAUSE_S = 600                 # nach einem Ausfall IRSA so lange ueberspringen
_irsa_down_until = 0.0

SEARCH_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_cache")

PIXEL_SCALE_ARCSEC = 6.15     # SPHEREx-Pixelgroesse am Himmel
MAX_CUTOUT_PIXELS = 400       # 400 px * 6.15" = 41' Kantenlaenge
MAX_RETRIES = 3               # wie im IRSA-Tutorial
EPOCH_GAP_DAYS = 30           # Luecke > 30 Tage = neuer Himmelsdurchgang (ca. alle 6 Monate)
MJD_J2000 = 51544.5

TRANSIENT_ERRORS = (
    TimeoutError,
    urllib.error.HTTPError,
    urllib.error.URLError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
)

# Bit-Nummern der FLAGS-Extension (aus dem IRSA "Data Overview"-Notebook)
FLAG_BITS = {
    0: "TRANSIENT", 1: "OVERFLOW", 2: "SUR_ERROR", 6: "NONFUNC", 7: "DICHROIC",
    9: "MISSING_DATA", 10: "HOT", 11: "COLD", 12: "FULLSAMPLE", 14: "PHANMISS",
    15: "NONLINEAR", 17: "PERSIST", 19: "OUTLIER", 20: "CROSSTALK", 21: "SOURCE",
    22: "GHOST", 23: "GHOST_FPA", 24: "GHOST_EXT", 25: "STREAK", 26: "BLOOM",
    27: "SNOWBALL", 28: "HALO",
}
BAD_BITS = (0, 1, 2, 6, 9, 10, 11, 14, 15, 19)
BAD_MASK = sum(1 << b for b in BAD_BITS)


# --- 3. SERVER ---
app = FastAPI(title="SPHEREx Grundsystem API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def start_page():
    """Liefert die Weboberflaeche aus (index.html muss im selben Ordner wie diese Datei liegen)."""
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(page):
        raise HTTPException(status_code=404, detail="index.html is missing from the folder of spherex_api.py.")
    return FileResponse(page)


# --- 4. HILFSFUNKTIONEN ---
SIMBAD_TAP_URLS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.harvard.edu/simbad/sim-tap/sync",
]

# Reserve fuer schnelle Testziele, falls SIMBAD nicht erreichbar ist (pmRA*cos(dec), pmDec in mas/Jahr)
KNOWN_PM = {
    "barnard's star": (-801.55, 10362.39),
    "barnards star": (-801.55, 10362.39),
    "gj 699": (-801.55, 10362.39),
    "eps ind b": (3961.40, -2538.30),
    "epsilon indi b": (3961.40, -2538.30),
    "wise 0855": (-8118.0, 680.0),
}


def _simbad_proper_motion(name: str, ra_deg: float, dec_deg: float):
    """Eigenbewegung (pmRA*cos(dec), pmDec) in mas/Jahr. Rueckgabe (Wert | None, Herkunft als Text)."""
    problems = []
    adql = (
        "SELECT main_id, ra, dec, pmra, pmdec FROM basic "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.7f}, {dec_deg:.7f}, 0.003)) = 1"
    )

    def nearest(rows):
        best = None
        for r_ra, r_dec, pmra, pmdec in rows:
            try:
                pmra, pmdec = float(pmra), float(pmdec)
                dist = np.hypot((float(r_ra) - ra_deg) * np.cos(np.radians(dec_deg)), float(r_dec) - dec_deg)
            except (TypeError, ValueError):
                continue
            if np.isfinite(pmra) and np.isfinite(pmdec) and (best is None or dist < best[0]):
                best = (dist, pmra, pmdec)
        return (best[1], best[2]) if best else None

    body = urllib.parse.urlencode({"request": "doQuery", "lang": "adql", "format": "json", "query": adql}).encode()
    for tap_url in SIMBAD_TAP_URLS:
        host = urllib.parse.urlparse(tap_url).netloc
        try:
            req = urllib.request.Request(tap_url, data=body, headers={"User-Agent": "spherex-bildvergleich/2.1 (NASA Space Apps)"})
            with urllib.request.urlopen(req, timeout=12) as r:
                payload = json.load(r)
            cols = [str(c.get("name", "")).lower() for c in payload.get("metadata", [])]
            i = {c: cols.index(c) for c in ("ra", "dec", "pmra", "pmdec")}
            pm = nearest([(row[i["ra"]], row[i["dec"]], row[i["pmra"]], row[i["pmdec"]]) for row in payload.get("data", [])])
            if pm:
                return pm, "SIMBAD" if "unistra" in host else f"SIMBAD mirror {host}"
            problems.append(f"{host}: kein Eintrag mit Eigenbewegung")
        except Exception as e:
            problems.append(f"{host}: {e!r}")

    try:
        from astroquery.simbad import Simbad
        try:
            t = Simbad.query_tap(adql)
            c = {x.lower(): x for x in t.colnames}
            pm = nearest([(row[c["ra"]], row[c["dec"]], row[c["pmra"]], row[c["pmdec"]]) for row in t])
            if pm:
                return pm, "SIMBAD"
            problems.append("astroquery-TAP: kein Eintrag mit Eigenbewegung")
        except Exception as e:
            problems.append(f"astroquery-TAP: {e!r}")
        simbad = Simbad()
        simbad.add_votable_fields("pmra", "pmdec")
        t = simbad.query_object(name)
        if t is not None and len(t) > 0:
            c = {x.lower(): x for x in t.colnames}
            pmra, pmdec = float(t[c["pmra"]][0]), float(t[c["pmdec"]][0])
            if np.isfinite(pmra) and np.isfinite(pmdec):
                return (pmra, pmdec), "SIMBAD"
        problems.append("astroquery-Name: kein Wert")
    except Exception as e:
        problems.append(f"astroquery: {e!r}")

    key = " ".join(name.lower().replace("\u2019", "'").split())
    if key in KNOWN_PM:
        print(f"[Hinweis] SIMBAD lieferte keine Eigenbewegung fuer '{name}' ({'; '.join(problems)}) - nutze eingebauten Gaia-DR3-Wert.")
        return KNOWN_PM[key], "built-in Gaia DR3 value (SIMBAD unreachable)"
    print(f"[Hinweis] Keine Eigenbewegung fuer '{name}': {'; '.join(problems)}")
    return None, "not found: " + "; ".join(problems)[:300]


def _propagate(ra_deg, dec_deg, pm, mjd, epoch_year=2000.0):
    """Verschiebt eine Position der Epoche `epoch_year` linear mit der Eigenbewegung zum Zeitpunkt `mjd`."""
    if pm is None:
        return ra_deg, dec_deg
    years = (mjd - (MJD_J2000 + (epoch_year - 2000.0) * 365.25)) / 365.25
    dec_new = dec_deg + pm[1] * years / 3.6e6
    ra_new = ra_deg + pm[0] * years / 3.6e6 / np.cos(np.radians(dec_deg))
    return ra_new % 360.0, dec_new


def resolve_target(target: str):
    """Eingabe -> (ra_deg, dec_deg, pm | None, wurde_per_name_aufgeloest, Herkunft der Eigenbewegung)"""
    text = target.strip()
    text = re.sub(r"(?<=[0-9sS])([+-])(?=\d+\s*[d:\u00b0])", r" \1", text) if " " not in text else text
    parts = text.replace(",", " ").split()

    if len(parts) == 2:
        try:
            ra, dec = float(parts[0]), float(parts[1])
            if 0 <= ra < 360 and -90 <= dec <= 90:
                return ra, dec, None, False, None
        except ValueError:
            pass

    try:
        c = SkyCoord(text, unit=(u.hourangle, u.deg), frame="icrs")
        return float(c.ra.deg), float(c.dec.deg), None, False, None
    except Exception:
        pass

    try:
        c = SkyCoord.from_name(text)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Target '{target}' is neither a coordinate nor a known object name.")
    ra, dec = float(c.ra.deg), float(c.dec.deg)
    if not USE_CATALOGS:
        return ra, dec, None, True, "none (catalogues disabled)"
    pm, source = _simbad_proper_motion(text, ra, dec)
    return ra, dec, pm, True, source


_S3_KEYS: dict[str, str] = {}          

_URL_PATTERNS = (
    r"^https://irsa\.ipac\.caltech\.edu/",
    r"^s3://nasa-irsa-spherex/",
    r"^https://nasa-irsa-spherex\.s3(\.[a-z0-9-]+)?\.amazonaws\.com/",
)


def _check_url(url: str):
    """Nur IRSA- und AWS-Adressen von SPHEREx zulassen (sonst koennte der Server beliebige Adressen laden)."""
    if not any(re.match(p, url) for p in _URL_PATTERNS) or ".." in url:
        raise HTTPException(status_code=400, detail="Image URLs must point to https://irsa.ipac.caltech.edu/ "
                            "or to the AWS bucket nasa-irsa-spherex.")


def _cloud_key(cloud_access) -> str | None:
    """S3-Schluessel aus der Spalte cloud_access der IRSA-Suche ({"aws": {"bucket_name", "key"}})."""
    if cloud_access is None:
        return None
    try:
        info = json.loads(cloud_access) if isinstance(cloud_access, (str, bytes)) else dict(cloud_access)
        aws = info.get("aws", {})
        if aws.get("bucket_name") == S3_BUCKET and aws.get("key"):
            return str(aws["key"]).lstrip("/")
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _s3_key_for(url: str) -> str | None:
    """Bildadresse -> S3-Schluessel im Bucket nasa-irsa-spherex, oder None."""
    if url in _S3_KEYS:
        return _S3_KEYS[url]
    base, _, query = url.partition("?")
    for pattern in (r"^https://irsa\.ipac\.caltech\.edu/ibe/data/spherex/(.+\.fits)$",
                    r"^s3://nasa-irsa-spherex/(.+\.fits)$",
                    r"^https://nasa-irsa-spherex\.s3(?:\.[a-z0-9-]+)?\.amazonaws\.com/(.+\.fits)$"):
        m = re.match(pattern, base)
        if m:
            return urllib.parse.unquote(m.group(1))
    # IRSA-Ausschnittdienst: .../ibe/cutout?...&path=spherex/<release>/level2/...fits
    path = urllib.parse.parse_qs(query).get("path", [""])[0]
    if path.startswith("spherex/") and path.endswith(".fits"):
        return path[len("spherex/"):]
    return None


def _section(hdu, y0, y1, x0, x1, dtype):
    """Nur den Ausschnitt [y0:y1, x0:x1] einer Bildebene lesen (Byte-Bereiche ueber S3)."""
    try:
        block = hdu.section[y0:y1, x0:x1]
    except Exception:                        
        block = hdu.data[y0:y1, x0:x1]
    return np.array(block, dtype=dtype)


@lru_cache(maxsize=400)
def _read_s3_cutout(key: str, ra: float, dec: float, size_deg: float):
    """
    Liest einen Ausschnitt direkt aus dem AWS-Bucket. Ergebnis wie _read_cutout (gleiche Schluessel),
    damit der Rest des Programms nicht merkt, woher die Daten kommen.

    1. Kopf der IMAGE-Ebene lesen, Zielposition im vollen Detektorbild bestimmen.
    2. Quadrat um das Ziel festlegen (gleiche Groesse, die beim IRSA-Dienst bestellt wurde).
    3. Nur dieses Quadrat aus IMAGE, ZODI, VARIANCE und FLAGS lesen.
    4. Wellenlaenge am Ziel aus dem spektralen WCS (Tabelle WCS-WAVE) im VOLLEN Bild bestimmen.
    5. Raeumliches WCS auf das Quadrat zuschneiden (verschiebt CRPIX und die SIP-Verzeichnung mit).
    """
    path = f"s3://{S3_BUCKET}/{key}"
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with fits.open(path, use_fsspec=True, fsspec_kwargs=S3_FSSPEC_KWARGS, lazy_load_hdus=True) as hdul:
                header = hdul["IMAGE"].header
                full_wcs = WCS(header)
                nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
                x_full, y_full = full_wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"))
                x_full, y_full = float(x_full), float(y_full)
                if not (np.isfinite(x_full) and np.isfinite(y_full)
                        and -0.5 <= x_full < nx - 0.5 and -0.5 <= y_full < ny - 0.5):
                    raise HTTPException(status_code=404, detail="Target coordinate is not on this detector image.")

                half = int(math.ceil(size_deg * 3600.0 / PIXEL_SCALE_ARCSEC / 2.0)) + 2
                xc, yc = int(round(x_full)), int(round(y_full))
                x0, x1 = max(xc - half, 0), min(xc + half + 1, nx)
                y0, y1 = max(yc - half, 0), min(yc + half + 1, ny)

                image = _section(hdul["IMAGE"], y0, y1, x0, x1, np.float32)        # MJy/sr
                zodi = _section(hdul["ZODI"], y0, y1, x0, x1, np.float32)          # MJy/sr
                variance = _section(hdul["VARIANCE"], y0, y1, x0, x1, np.float32)  # (MJy/sr)^2
                flags = _section(hdul["FLAGS"], y0, y1, x0, x1, np.int64)

                spectral_wcs = WCS(header=header, fobj=hdul, key="W")
                spectral_wcs.sip = None
                wave, bandwidth = spectral_wcs.pixel_to_world(x_full, y_full)

                return {
                    "image": image, "zodi": zodi, "variance": variance, "flags": flags,
                    "wcs": full_wcs[y0:y1, x0:x1], "x": x_full - x0, "y": y_full - y0,
                    "wave_um": float(wave.to_value(u.um)), "bw_um": float(bandwidth.to_value(u.um)),
                    "detector": header.get("DETECTOR"),
                    "version": hdul[0].header.get("VERSION", ""),
                }
        except HTTPException:
            raise
        except FileNotFoundError:
            raise                                
        except Exception as e:
            last_error = e
            print(f"[Hinweis] AWS-Abruf fehlgeschlagen (Versuch {attempt + 1}/{MAX_RETRIES}): {e!r}\n          {path}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"AWS-Bucket nicht erreichbar nach {MAX_RETRIES} Versuchen: {last_error!r}")


def _read_frame(fits_url: str, ra: float, dec: float, size_deg: float):
    """Holt einen Ausschnitt: je nach SPHEREX_IMAGE_SOURCE aus AWS, vom IRSA-Dienst oder erst AWS, dann IRSA."""
    _check_url(fits_url)
    key = _s3_key_for(fits_url)
    is_irsa = fits_url.startswith("https://irsa.ipac.caltech.edu/")
    if IMAGE_SOURCE in ("aws", "auto") and key:
        try:
            return _read_s3_cutout(key, round(ra, 7), round(dec, 7), round(size_deg, 5))
        except HTTPException:
            raise
        except Exception as e:
            if IMAGE_SOURCE == "aws" or not is_irsa:
                if isinstance(e, FileNotFoundError):
                    raise HTTPException(status_code=404, detail=f"File not found in the AWS bucket: {key}")
                raise HTTPException(status_code=502, detail=f"AWS bucket unreachable: {e}")
            print(f"[Hinweis] AWS fehlgeschlagen ({e!r}) - versuche den IRSA-Ausschnittdienst.")
    if is_irsa and IMAGE_SOURCE in ("irsa", "auto"):
        return _read_cutout(_cutout_url(fits_url, ra, dec, size_deg), ra, dec)
    raise HTTPException(status_code=400, detail="No S3 key for this URL "
                        "(check the URL or set SPHEREX_IMAGE_SOURCE=auto).")


def _search_cache_path(ra: float, dec: float) -> str:
    """Datei im Zwischenspeicher je Himmelsposition (auf 0,001 Grad = 3,6" gerundet) und Release-Liste."""
    name = f"{ra:.3f}_{dec:+.3f}_{','.join(sorted(COLLECTIONS))}"
    return os.path.join(SEARCH_CACHE_DIR, hashlib.sha1(name.encode()).hexdigest()[:16] + ".json")


_CACHE_COLUMNS = ["access_url", "t_min", "t_max", "energy_bandpassname", "em_min", "em_max", "t_exptime",
                  "obs_id", "obs_collection", "dataproduct_type", "cloud_access"]


def _save_search_cache(ra: float, dec: float, df):
    try:
        os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
        cols = [c for c in _CACHE_COLUMNS if c in df.columns]
        payload = {"ra": ra, "dec": dec, "collections": COLLECTIONS, "saved": Time.now().isot,
                   "rows": json.loads(df[cols].to_json(orient="records"))}
        tmp = _search_cache_path(ra, dec) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _search_cache_path(ra, dec))
    except Exception as e:
        print(f"[Hinweis] Suchergebnis konnte nicht zwischengespeichert werden: {e!r}")


def _load_search_cache(ra: float, dec: float):
    """Gespeichertes Suchergebnis als DataFrame plus Speicherdatum, oder (None, None)."""
    import pandas as pd
    path = _search_cache_path(ra, dec)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return pd.DataFrame(payload["rows"]), payload.get("saved", "?")


def _cutout_url(access_url: str, ra: float, dec: float, size_deg: float) -> str:
    """IRSA schneidet serverseitig aus, wenn man center/size anhaengt."""
    sep = "&" if "?" in access_url else "?"
    return f"{access_url}{sep}center={ra:.7f},{dec:.7f}d&size={size_deg:.5f}"


def _flag_names(value: int):
    return [name for bit, name in FLAG_BITS.items() if value & (1 << bit)]


def _north_up(data, src_wcs, ra, dec, npix, order_nearest=False):
    """
    Legt `data` auf ein nord-oben/ost-links TAN-Raster (npix x npix, 6.15"/px), zentriert auf (ra, dec).

    WICHTIGE KORREKTUR (v2.1):
    Standard ist jetzt bilineare Interpolation (`order_nearest=False`) mit NaN-Gewichtung statt
    hartem Runden (`np.rint`). Dadurch entfallen die künstlichen 0.5-Pixel-Rundungssprünge (ca. 3"),
    die bei Differenzbildern und Stacks sonst an jedem unbewegten Stern starke Dipol-Artefakte erzeugen.
    """
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out_wcs.wcs.crval = [ra, dec]
    out_wcs.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]
    out_wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600.0, PIXEL_SCALE_ARCSEC / 3600.0]

    yy, xx = np.indices((npix, npix))
    ra_grid, dec_grid = out_wcs.wcs_pix2world(xx, yy, 0)
    sx, sy = src_wcs.all_world2pix(ra_grid, dec_grid, 0, quiet=True)

    ny, nx = data.shape
    if order_nearest or nx < 2 or ny < 2:
        ix = np.rint(sx)
        iy = np.rint(sy)
        inside = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
        ix = np.where(inside, ix, 0).astype(int)
        iy = np.where(inside, iy, 0).astype(int)
        return data[iy, ix], inside

    inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0) & (sx <= nx - 1) & (sy <= ny - 1)
    x0 = np.clip(np.floor(np.where(inside, sx, 0)).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(np.where(inside, sy, 0)).astype(int), 0, ny - 2)
    wx = np.where(inside, sx - x0, 0.0)
    wy = np.where(inside, sy - y0, 0.0)

    weights = [(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy]
    vals = [data[y0, x0], data[y0, x0 + 1], data[y0 + 1, x0], data[y0 + 1, x0 + 1]]

    num = np.zeros((npix, npix), dtype=np.float32)
    den = np.zeros((npix, npix), dtype=np.float32)
    for w, v in zip(weights, vals):
        ok = inside & np.isfinite(v)
        num += np.where(ok, w * v, 0.0)
        den += np.where(ok, w, 0.0)

    valid_interp = inside & (den >= 0.25)
    out = np.where(valid_interp, num / np.maximum(den, 1e-9), np.nan)
    return out.astype(np.float32), inside


@lru_cache(maxsize=400)
def _read_cutout(url: str, ra: float, dec: float):
    """Oeffnet ein Cutout-MEF und holt alles heraus, was wir brauchen (mit Retry)."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with fits.open(url, memmap=False, cache=False, show_progress=False) as hdul:
                header = hdul["IMAGE"].header
                image = np.asarray(hdul["IMAGE"].data, dtype=np.float32)
                zodi = np.asarray(hdul["ZODI"].data, dtype=np.float32)
                variance = np.asarray(hdul["VARIANCE"].data, dtype=np.float32)
                flags = np.asarray(hdul["FLAGS"].data).astype(np.int64)

                spatial_wcs = WCS(header)
                x_pix, y_pix = spatial_wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"))
                x_pix, y_pix = float(x_pix), float(y_pix)

                spectral_wcs = WCS(header=header, fobj=hdul, key="W")
                spectral_wcs.sip = None
                wave, bandwidth = spectral_wcs.pixel_to_world(x_pix, y_pix)
                wave_um = float(wave.to_value(u.um))
                bw_um = float(bandwidth.to_value(u.um))

                return {
                    "image": image, "zodi": zodi, "variance": variance, "flags": flags,
                    "wcs": spatial_wcs, "x": x_pix, "y": y_pix,
                    "wave_um": wave_um, "bw_um": bw_um,
                    "detector": header.get("DETECTOR"),
                    "version": hdul[0].header.get("VERSION", ""),
                }
        except TRANSIENT_ERRORS as e:
            last_error = e
            code = getattr(e, "code", None)
            print(f"[Hinweis] Abruf fehlgeschlagen (Versuch {attempt + 1}/{MAX_RETRIES}): {e!r}\n          {url}")
            if code is not None and 400 <= code < 500 and code != 429:
                raise HTTPException(status_code=404, detail=f"IRSA returns no cutout for this image (HTTP {code}).")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3 * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"IRSA unreachable after {MAX_RETRIES} attempts: {last_error!r}")


N_RING = 360


def _sigma_clipped(values, n_iter=3, k=3.0):
    """Robuster Median und Streuung (1.4826 * MAD) nach iterativem k-sigma-Clipping."""
    v = values[np.isfinite(values)]
    for _ in range(n_iter):
        if v.size < 10:
            break
        med = np.median(v)
        sig = 1.4826 * np.median(np.abs(v - med))
        keep = np.abs(v - med) < k * sig
        if sig <= 0 or keep.all():
            break
        v = v[keep]
    if v.size == 0:
        return None
    med = float(np.median(v))
    return med, max(1.4826 * float(np.median(np.abs(v - med))), 1e-9), int(v.size)


def _local_sky(clean, xi, yi):
    """Hintergrund B, Streuung sigma (MJy/sr) und Zahl der benutzten Ringpixel; None am Bildrand."""
    ny, nx = clean.shape
    if not (10 <= xi < nx - 10 and 10 <= yi < ny - 10):
        return None
    box = clean[yi - 10:yi + 11, xi - 10:xi + 11]
    yy, xx = np.indices(box.shape)
    ring = box[np.maximum(np.abs(yy - 10), np.abs(xx - 10)) >= 5]
    if np.isfinite(ring).sum() < 50:
        return None
    return _sigma_clipped(ring)


def _find_source(clean, variance, x, y, radius=6, min_snr=6.0):
    """
    Hellste Quelle im Umkreis von `radius` Pixeln (6 px = 37") um die Sollposition.
    Der groessere Suchradius stellt sicher, dass schnelle Objekte auch über mehrere Epochen
    im Suchfenster bleiben.
    """
    xi, yi = int(round(x)), int(round(y))
    ny, nx = clean.shape
    if not (radius + 2 <= xi < nx - (radius + 2) and radius + 2 <= yi < ny - (radius + 2)):
        return None
    sky_noise = _local_sky(clean, xi, yi)
    if sky_noise is None:
        return None
    sky, noise, _ = sky_noise
    win = clean[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]
    if not np.isfinite(win).any():
        return None
    py, px = np.unravel_index(np.nanargmax(win), win.shape)
    if px in (0, 2 * radius) or py in (0, 2 * radius):
        return None
    px, py = xi - radius + int(px), yi - radius + int(py)
    snr = (float(clean[py, px]) - sky) / noise
    if snr < min_snr:
        return None
    stamp = clean[py - 2:py + 3, px - 2:px + 3] - sky
    use = np.isfinite(stamp) & (stamp > 0)
    weight = np.where(use, stamp, 0.0)
    total = float(weight.sum())
    if total <= 0:
        return None
    var = variance[py - 2:py + 3, px - 2:px + 3]
    var = np.where(np.isfinite(var) & (var > 0), var, noise ** 2)
    yy, xx = np.indices(weight.shape)
    cx = float((weight * xx).sum() / total)
    cy = float((weight * yy).sum() / total)
    ex = float(np.sqrt((np.where(use, var, 0.0) * (xx - cx) ** 2).sum()) / total)
    ey = float(np.sqrt((np.where(use, var, 0.0) * (yy - cy) ** 2).sum()) / total)
    return {"x": px - 2 + cx, "y": py - 2 + cy, "ex": ex, "ey": ey, "snr": snr}


def _find_saturated(clean, variance, x, y, radius=6, reach=7):
    """Position eines UEBERBELICHTETEN Sterns aus seinen symmetrischen Fluegeln."""
    xi, yi = int(round(x)), int(round(y))
    ny, nx = clean.shape
    margin = radius + reach + 2
    if not (margin <= xi < nx - margin and margin <= yi < ny - margin):
        return None
    win = clean[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]
    wy, wx = np.indices(win.shape)
    hole = ~np.isfinite(win) & ((wx - radius) ** 2 + (wy - radius) ** 2 <= radius ** 2)
    if hole.sum() < 1:
        return None
    cx = xi - radius + float(wx[hole].mean())
    cy = yi - radius + float(wy[hole].mean())

    sky_noise = _local_sky(clean, xi, yi)
    sky, noise = (sky_noise[0], sky_noise[1]) if sky_noise else (float(np.nanmedian(clean)), 1e-9)
    dy, dx = np.indices((2 * reach + 1, 2 * reach + 1)) - reach
    circle = dx ** 2 + dy ** 2 <= reach ** 2
    ex = ey = snr = None
    for _ in range(12):
        x0, y0 = int(round(cx)), int(round(cy))
        if not (reach <= x0 < nx - reach and reach <= y0 < ny - reach):
            return None
        sub = clean[y0 - reach:y0 + reach + 1, x0 - reach:x0 + reach + 1] - sky
        ok = np.isfinite(sub) & np.isfinite(sub[::-1, ::-1]) & circle
        weight = np.where(ok, np.clip(np.nan_to_num(sub), 0, None), 0.0)
        total = float(weight.sum())
        if total <= 0 or ok.sum() < 12:
            return None
        new_x = x0 + float((weight * dx).sum() / total)
        new_y = y0 + float((weight * dy).sum() / total)
        var = variance[y0 - reach:y0 + reach + 1, x0 - reach:x0 + reach + 1]
        var = np.where(ok & np.isfinite(var) & (var > 0), var, np.where(ok, noise ** 2, 0.0))
        ex = float(np.sqrt((var * (dx - (new_x - x0)) ** 2).sum()) / total)
        ey = float(np.sqrt((var * (dy - (new_y - y0)) ** 2).sum()) / total)
        snr = total / (noise * np.sqrt(ok.sum()))
        moved = np.hypot(new_x - cx, new_y - cy)
        cx, cy = new_x, new_y
        if moved < 0.01:
            break
    if snr is None or snr < 20:
        return None
    if np.hypot(cx - x, cy - y) > radius:
        return None
    return {"x": cx, "y": cy, "ex": max(ex, 0.25), "ey": max(ey, 0.25), "snr": float(snr), "saturated": True}


def _aperture_flux_mjy(clean, variance, wcs, x, y):
    """Aperturphotometrie 3x3 Pixel an (x, y). Rueckgabe (Fluss, 1-sigma-Fehler) in mJy oder (None, None)."""
    xi, yi = int(round(x)), int(round(y))
    sky_noise = _local_sky(clean, xi, yi)
    if sky_noise is None:
        return None, None
    sky, noise, n_ring = sky_noise
    core = clean[yi - 1:yi + 2, xi - 1:xi + 2]
    if not np.all(np.isfinite(core)):
        return None, None
    var = variance[yi - 1:yi + 2, xi - 1:xi + 2]
    var = np.where(np.isfinite(var) & (var > 0), var, noise ** 2)
    var_background = (1.2533 * noise) ** 2 / n_ring
    flux_var = float(var.sum()) + core.size ** 2 * var_background
    to_mjy = float(proj_plane_pixel_area(wcs)) * (np.pi / 180.0) ** 2 * 1e9
    return float((core - sky).sum() * to_mjy), float(np.sqrt(flux_var) * to_mjy)


def _cutout_size_deg(npix: int) -> float:
    return max(npix * PIXEL_SCALE_ARCSEC / 3600.0 * 1.5, 0.08)


def _render_png(frame, valid, vmin=None, vmax=None):
    """Macht aus einem Messwert-Raster ein PNG (Graustufen + Transparenz)."""
    if vmin is None or vmax is None:
        values = frame[valid]
        sky = float(np.median(values))
        noise = 1.4826 * float(np.median(np.abs(values - sky))) or 1e-6
        auto_min = sky - 1.0 * noise
        auto_max = max(float(np.percentile(values, 99.8)), sky + 25.0 * noise)
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    norm = np.clip((np.where(valid, frame, vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    norm = np.arcsinh(norm * 12.0) / np.arcsinh(12.0)
    gray = np.flipud((norm * 255).astype(np.uint8))
    alpha = np.flipud(np.where(valid, 255, 0).astype(np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(np.dstack([gray, alpha]), mode="LA").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), float(vmin), float(vmax)


def _north_up_frame(fits_url: str, ra: float, dec: float, npix: int, subtract_zodi: bool = True):
    """Bereinigtes, bilinear nach Norden ausgerichtetes Messwert-Raster eines Bildes plus Wellenlaenge."""
    d = _read_frame(fits_url, ra, dec, _cutout_size_deg(npix))
    clean = d["image"] - d["zodi"] if subtract_zodi else d["image"]
    clean = np.where((d["flags"] & BAD_MASK) == 0, clean, np.nan)
    frame, inside = _north_up(clean, d["wcs"], ra, dec, npix, order_nearest=False)
    return np.where(inside, frame, np.nan).astype(np.float32), d["wave_um"]


@lru_cache(maxsize=512)
def _build_frame(fits_url: str, ra: float, dec: float, npix: int, subtract_zodi: bool,
                 vmin: float | None, vmax: float | None):
    """Kompletter Weg von der FITS-URL zum fertigen Einzel-Frame."""
    d = _read_frame(fits_url, ra, dec, _cutout_size_deg(npix))

    ny, nx = d["image"].shape
    xi, yi = int(round(d["x"])), int(round(d["y"]))
    if not (0 <= xi < nx and 0 <= yi < ny):
        raise HTTPException(status_code=404, detail="Target coordinate is not on this detector image.")

    clean = d["image"] - d["zodi"] if subtract_zodi else d["image"]
    clean = np.where((d["flags"] & BAD_MASK) == 0, clean, np.nan)

    frame, inside = _north_up(clean, d["wcs"], ra, dec, npix, order_nearest=False)
    valid = inside & np.isfinite(frame)
    if valid.sum() < 10:
        raise HTTPException(status_code=404, detail="No valid pixel in the cutout.")

    image_b64, vmin, vmax = _render_png(frame, valid, vmin, vmax)

    target_flag = int(d["flags"][yi, xi])
    pixel_value = float(clean[yi, xi])
    sigma = float(np.sqrt(d["variance"][yi, xi])) if np.isfinite(d["variance"][yi, xi]) else None

    src = _find_saturated(clean, d["variance"], d["x"], d["y"])
    saturated = src is not None
    if not saturated:
        src = _find_source(clean, d["variance"], d["x"], d["y"])
    east = north = pos_err = None
    if src:
        scale = float(np.sqrt(proj_plane_pixel_area(d["wcs"]))) * 3600.0
        pos_err = scale * float(np.sqrt((src["ex"] ** 2 + src["ey"] ** 2) / 2.0))
        c = d["wcs"].pixel_to_world(src["x"], src["y"])
        east = ((float(c.ra.deg) - ra + 540.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * 3600.0
        north = (float(c.dec.deg) - dec) * 3600.0

    mx, my = (src["x"], src["y"]) if src else (d["x"], d["y"])
    flux_mjy, flux_err = _aperture_flux_mjy(clean, d["variance"], d["wcs"], mx, my)
    if saturated:
        flux_mjy = flux_err = None

    return {
        "image_b64": image_b64,
        "detector": d["detector"],
        "pipeline_version": d["version"],
        "pixel_x": round(d["x"], 1),
        "pixel_y": round(d["y"], 1),
        "exact_wavelength_um": round(d["wave_um"], 4),
        "exact_bandwidth_um": round(d["bw_um"], 4),
        "flux_at_target_mjy_sr": round(pixel_value, 3) if np.isfinite(pixel_value) else None,
        "sigma_at_target_mjy_sr": round(sigma, 3) if sigma is not None else None,
        "aperture_flux_mjy": round(flux_mjy, 5) if flux_mjy is not None else None,
        "aperture_flux_err_mjy": round(flux_err, 5) if flux_err is not None else None,
        "source_detected": src is not None,
        "saturated": saturated,
        "source_snr": round(src["snr"], 1) if src else None,
        "offset_east_arcsec": round(east, 3) if east is not None else None,
        "offset_north_arcsec": round(north, 3) if north is not None else None,
        "offset_err_arcsec": round(pos_err, 3) if pos_err is not None else None,
        "flags_at_target": _flag_names(target_flag),
        "target_pixel_bad": bool(target_flag & BAD_MASK),
        "coverage": round(float(inside.mean()), 3),
        "bad_pixel_fraction": round(float((inside & ~valid).mean()), 3),
        "vmin": float(vmin), "vmax": float(vmax),
        "size_px": npix,
        "fov_arcmin": round(npix * PIXEL_SCALE_ARCSEC / 60.0, 2),
        "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        "orientation": "N oben, O links",
    }


DETECTOR_RANGE_M = {1: (0.75e-6, 1.10e-6), 2: (1.10e-6, 1.65e-6), 3: (1.63e-6, 2.43e-6),
                    4: (2.42e-6, 3.82e-6), 5: (3.81e-6, 4.43e-6), 6: (4.41e-6, 5.01e-6)}


def _search_tap(ra: float, dec: float):
    """Reserve-Suche ueber den TAP-Dienst von IRSA, falls der SIA-Dienst ausfaellt."""
    import pandas as pd
    adql = (
        "SELECT a.uri, p.* FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
        f"WHERE 1 = CONTAINS(POINT('ICRS', {ra:.7f}, {dec:.7f}), p.poly) ORDER BY p.time_bounds_lower"
    )
    raw = Irsa.query_tap(adql).to_table().to_pandas()
    col = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in col:
                return raw[col[n]]
        return None

    uri = pick("uri").astype(str)
    df = pd.DataFrame({"access_url": "https://irsa.ipac.caltech.edu/" + uri.str.lstrip("/")})
    df["t_min"] = pd.to_numeric(pick("time_bounds_lower"), errors="coerce")
    t_max = pick("time_bounds_upper")
    df["t_max"] = pd.to_numeric(t_max, errors="coerce") if t_max is not None else df["t_min"]
    df["energy_bandpassname"] = pick("energy_bandpassname").astype(str)
    exposure = pick("time_exposure", "t_exptime")
    df["t_exptime"] = pd.to_numeric(exposure, errors="coerce") if exposure is not None else 0.0
    det = df["energy_bandpassname"].str.extract(r"(\d)\s*$")[0].astype(float)
    lo, hi = pick("energy_bounds_lower", "em_min"), pick("energy_bounds_upper", "em_max")
    df["em_min"] = pd.to_numeric(lo, errors="coerce") if lo is not None else det.map(lambda d: DETECTOR_RANGE_M.get(d, (np.nan, np.nan))[0])
    df["em_max"] = pd.to_numeric(hi, errors="coerce") if hi is not None else det.map(lambda d: DETECTOR_RANGE_M.get(d, (np.nan, np.nan))[1])
    obs = pick("obsid", "obs_id", "observationid")
    df["obs_id"] = obs.astype(str) if obs is not None else uri.str.extract(r"([^/]+)\.fits")[0]
    df["obs_collection"] = "spherex_" + uri.str.extract(r"spherex/([^/]+)/")[0].fillna("")
    cloud = pick("cloud_access")
    if cloud is not None:
        df["cloud_access"] = cloud.astype(str)
    wanted = df[df["obs_collection"].isin(COLLECTIONS)]
    return wanted if len(wanted) else df


# --- 5. ENDPUNKT 1: SUCHE ---
@app.get("/api/search")
def search_spherex(
    target: str,
    detector: int | None = Query(None, ge=1, le=6, description="Nur dieser Detektor (1-6)"),
    limit: int = Query(2000, ge=1, le=10000),
    pmra: float | None = Query(None, description="Eigenbewegung in RA (mu_alpha*cos(dec)) in mas/Jahr; ueberschreibt SIMBAD"),
    pmdec: float | None = Query(None, description="Eigenbewegung in Dec in mas/Jahr"),
    epoch: float = Query(2000.0, ge=1900, le=2100, description="Epoche (Jahr), fuer die die Koordinaten gelten"),
):
    ra0, dec0, pm, by_name, pm_source = resolve_target(target)
    if pmra is not None and pmdec is not None:
        pm, pm_source = (pmra, pmdec), "user input"
    if not by_name and pm is None:
        pm_source = "none (coordinates without proper motion)"

    ra, dec = _propagate(ra0, dec0, pm, Time.now().mjd, epoch)

    global _irsa_down_until
    df, last_error, service = None, None, None
    ask_irsa = SEARCH_MODE == "irsa" or (SEARCH_MODE == "auto" and time.time() >= _irsa_down_until)
    if ask_irsa:
        for attempt in range(MAX_RETRIES):
            try:
                results = Irsa.query_sia(
                    pos=(SkyCoord(ra, dec, unit="deg", frame="icrs"), 1 * u.arcsec),
                    collection=COLLECTIONS,
                )
                df = results.to_pandas()
                service = "IRSA SIA"
                break
            except Exception as e:
                last_error = e
                print(f"[Hinweis] IRSA-Suche fehlgeschlagen (Versuch {attempt + 1}/{MAX_RETRIES}): {str(e)[:200]}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(4 * (attempt + 1))
        if df is None:
            try:
                df = _search_tap(ra, dec)
                service = "IRSA TAP (fallback, SIA unreachable)"
                print(f"[Hinweis] SIA nicht erreichbar, Suche ueber TAP erfolgreich ({len(df)} Aufnahmen).")
            except Exception as e:
                last_error = last_error or e
                print(f"[Hinweis] Auch die TAP-Suche schlug fehl: {str(e)[:200]}")
                _irsa_down_until = time.time() + IRSA_PAUSE_S
                print(f"[Hinweis] IRSA wird fuer {IRSA_PAUSE_S // 60} Minuten uebersprungen.")
    elif SEARCH_MODE == "auto":
        print("[Hinweis] IRSA war vor Kurzem nicht erreichbar - direkt Zwischenspeicher / AWS-Suche.")

    if df is None and SEARCH_MODE == "irsa":
        raise HTTPException(status_code=502, detail="The NASA archive IRSA is not responding (SIA and TAP tried). "
                            "With SPHEREX_SEARCH=auto or aws the server searches the AWS bucket itself. "
                            "Technical message: " + str(last_error)[:300])
    if df is None and SEARCH_MODE == "auto":
        cached, saved = _load_search_cache(ra, dec)
        if cached is not None:
            df = cached
            service = f"cached result from {saved[:10]} (IRSA unreachable)"
            print(f"[Hinweis] Nutze gespeichertes Suchergebnis ({len(df)} Aufnahmen).")
    if df is None:
        try:
            t_s3 = time.time()
            df = spherex_s3_search.search_dataframe(ra, dec, COLLECTIONS)
            service = f"own search in the AWS bucket ({time.time() - t_s3:.0f} s)"
        except Exception as e:
            print(f"[Hinweis] AWS-Suche fehlgeschlagen: {e!r}")
            raise HTTPException(status_code=502, detail="Neither IRSA nor the own search in the AWS bucket worked. "
                                f"AWS message: {str(e)[:300]}" + (f" IRSA message: {str(last_error)[:200]}" if last_error else ""))
    if service.startswith(("IRSA", "own")) and len(df):
        _save_search_cache(ra, dec, df)

    base = {"ra": round(ra, 6), "dec": round(dec, 6), "collections": COLLECTIONS,
            "proper_motion_mas_yr": list(pm) if pm else None, "proper_motion_source": pm_source,
            "input_ra": round(ra0, 6), "input_dec": round(dec0, 6), "input_epoch": epoch, "search_service": service,
            "catalogs_used": USE_CATALOGS and by_name, "image_source": IMAGE_SOURCE}
    if len(df) == 0:
        return {**base, "count": 0, "epochs": 0, "images": []}

    df = df[df["t_min"].notna() & df["access_url"].notna()].copy()
    if "dataproduct_type" in df.columns:
        df = df[df["dataproduct_type"].astype(str).str.lower() == "image"]
    if "obs_id" in df.columns:
        df = df.drop_duplicates(subset=["obs_id", "energy_bandpassname"], keep="last")
    df = df.sort_values("t_min").reset_index(drop=True)

    df["detector"] = df["energy_bandpassname"].astype(str).str.extract(r"(\d)\s*$")[0].astype(float)
    df["epoch"] = (df["t_min"].diff() > EPOCH_GAP_DAYS).cumsum() + 1
    n_total, n_epochs = len(df), int(df["epoch"].max()) if len(df) else 0

    if detector is not None:
        df = df[df["detector"] == detector]
    df = df.head(limit)
    if len(df) == 0:
        return {**base, "count": 0, "total": n_total, "epochs": n_epochs, "images": []}

    if pm:
        ra, dec = _propagate(ra0, dec0, pm, float(df["t_min"].median()), epoch)
        base.update(ra=round(ra, 6), dec=round(dec, 6))

    if "cloud_access" in df.columns:
        for url, cloud in zip(df["access_url"].astype(str), df["cloud_access"]):
            key = _cloud_key(cloud)
            if key:
                _S3_KEYS[url] = key

    t_mid = (df["t_min"] + df["t_max"].fillna(df["t_min"])) / 2.0
    iso = Time(t_mid.values, format="mjd").isot
    t0 = float(df["t_min"].iloc[0])

    images = []
    for i, (_, row) in enumerate(df.iterrows()):
        images.append({
            "obs_id": str(row.get("obs_id", "")),
            "collection": str(row.get("obs_collection", "")),
            "date_iso": str(iso[i]),
            "mjd": round(float(t_mid.iloc[i]), 5),
            "days_since_first": round(float(row["t_min"]) - t0, 3),
            "epoch": int(row["epoch"]),
            "band": str(row.get("energy_bandpassname", "")),
            "detector": int(row["detector"]) if np.isfinite(row["detector"]) else None,
            "wave_approx_um": round((float(row["em_min"]) + float(row["em_max"])) / 2.0 * 1e6, 3),
            "exptime_s": round(float(row.get("t_exptime", 0) or 0), 1),
            "access_url": str(row["access_url"]),
            "s3_key": _S3_KEYS.get(str(row["access_url"])) or _s3_key_for(str(row["access_url"])),
        })

    return {
        **base,
        "resolved_by_name": by_name,
        "count": len(images),
        "total": n_total,
        "epochs": n_epochs,
        "baseline_days": round(float(df["t_min"].iloc[-1]) - t0, 2),
        "images": images,
    }


# --- 6. ENDPUNKT 2: EIN FRAME ---
@app.get("/api/load_spherex_mef")
def load_spherex_mef(
    fits_url: str,
    ra: float = Query(..., ge=0, lt=360),
    dec: float = Query(..., ge=-90, le=90),
    cutout_pixels: int = Query(100, ge=16, le=MAX_CUTOUT_PIXELS),
    subtract_zodi: bool = True,
    vmin: float | None = Query(None, description="Fester Schwarzpunkt in MJy/sr"),
    vmax: float | None = Query(None, description="Fester Weisspunkt in MJy/sr"),
):
    _check_url(fits_url)
    try:
        return _build_frame(fits_url, round(ra, 7), round(dec, 7), cutout_pixels, subtract_zodi, vmin, vmax)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the SPHEREx file: {e}")


# --- 6b. ENDPUNKT 3: VIELE AUFNAHMEN ZU EINEM BILD ZUSAMMENRECHNEN (Zeitraffer) ---
class StackRequest(BaseModel):
    ra: float
    dec: float
    urls: list[str]
    cutout_pixels: int = 100
    subtract_zodi: bool = True
    vmin: float | None = None
    vmax: float | None = None
    lam_min: float | None = None
    lam_max: float | None = None


MAX_STACK = 120


@app.post("/api/stack")
def stack_frames(req: StackRequest):
    """
    Rechnet viele Aufnahmen derselben Stelle zu EINEM Bild zusammen.

    WICHTIGE KORREKTUR (v2.1):
    Statt eines reinen Medians (der schmalbandige Quellen wie Methan-Zwerge ausloescht, wenn sie
    in >50% des Detektor-Wellenlaengenbereichs dunkel sind) nutzen wir ein Sigma-Clipped Mean:
    Einzelne Ausreisser nach oben/unten (> 4 MAD, z.B. kosmische Strahlung) werden verworfen,
    danach wird das arithmetische Mittel gebildet.
    """
    if not req.urls:
        raise HTTPException(status_code=400, detail="No exposures given.")
    if len(req.urls) > MAX_STACK:
        raise HTTPException(status_code=400, detail=f"At most {MAX_STACK} exposures per image.")
    if not (16 <= req.cutout_pixels <= 200):
        raise HTTPException(status_code=400, detail="cutout_pixels must be between 16 and 200 for averaged images.")
    for url in req.urls:
        _check_url(url)
    ra, dec = round(req.ra, 7), round(req.dec, 7)

    def load(url):
        try:
            frame, wave = _north_up_frame(url, ra, dec, req.cutout_pixels, req.subtract_zodi)
        except Exception as e:
            print(f"[Hinweis] Aufnahme fuer das gemittelte Bild uebersprungen: {getattr(e, 'detail', e)!r}")
            return None
        if req.lam_min is not None and wave < req.lam_min:
            return None
        if req.lam_max is not None and wave > req.lam_max:
            return None
        finite = np.isfinite(frame)
        if finite.sum() < 10:
            return None
        return frame - np.median(frame[finite]), wave

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = [r for r in pool.map(load, req.urls) if r is not None]
    if not loaded:
        raise HTTPException(status_code=404, detail="None of the exposures was usable.")

    cube = np.stack([r[0] for r in loaded])
    waves = np.array([r[1] for r in loaded])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if len(loaded) >= 4:
            med = np.nanmedian(cube, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(cube - med), axis=0)
            keep = np.abs(cube - med) <= np.maximum(4.0 * mad, 1e-5)
            image = np.nanmean(np.where(keep, cube, np.nan), axis=0)
        else:
            image = np.nanmean(cube, axis=0)
    valid = np.isfinite(image)
    if valid.sum() < 10:
        raise HTTPException(status_code=404, detail="No valid pixel in the averaged image.")
    image_b64, vmin, vmax = _render_png(image, valid, req.vmin, req.vmax)
    return {
        "image_b64": image_b64,
        "n_used": len(loaded),
        "n_requested": len(req.urls),
        "wavelength_min_um": round(float(waves.min()), 3),
        "wavelength_max_um": round(float(waves.max()), 3),
        "vmin": vmin, "vmax": vmax,
        "size_px": req.cutout_pixels,
        "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        "coverage": round(float(valid.mean()), 3),
    }


# --- 6c. ENDPUNKT 4: BEWEGUNGSANZEIGE DURCH DIFFERENZBILDER (Moving Target Indication) ---
def _mti_combine(frames_a, waves_a, frames_b, waves_b, max_dlam):
    """
    Paart Aufnahmen zweier Durchgaenge nach Wellenlaenge, gleicht den globalen Flussfaktor an
    und bildet das Differenzbild.

    WICHTIGE KORREKTUR (v2.1):
    Als Referenz fuer die Unterdrueckung von Stern-Residuen wird NICHT mehr `0.5 * (A + B)`
    verwendet (das enthielt auch das bewegte Objekt selbst und hat dessen Signal ausgeloescht!),
    sondern das pixelweise Minimum `min(max(A, 0), max(B, 0))`:
      - Ein unbewegter Stern ist in A UND B am selben Ort hell -> min(A, B) ist gross -> Residuen werden gedaempft.
      - Ein bewegtes Objekt ist am alten Ort nur in A und am neuen Ort nur in B hell -> min(A, B) ~ 0 -> KEINE Daempfung!
    """
    candidates = sorted((abs(wa - wb), i, j) for i, wa in enumerate(waves_a) for j, wb in enumerate(waves_b))
    used_a, used_b, pairs = set(), set(), []
    for dlam, i, j in candidates:
        if dlam > max_dlam:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((i, j, dlam))
    if not pairs:
        return None

    diffs, scales = [], []
    for i, j, _ in pairs:
        a, b = frames_a[i], frames_b[j]
        both = np.isfinite(a) & np.isfinite(b)
        scale = 1.0
        if both.sum() > 50:
            noise = 1.4826 * np.median(np.abs(a[both] - np.median(a[both]))) or 1e-9
            bright = both & (a > 5 * noise)
            if bright.sum() >= 10:
                scale = float((a[bright] * b[bright]).sum() / (a[bright] ** 2).sum())
                scale = min(max(scale, 0.5), 2.0)
        diffs.append(b - scale * a)
        scales.append(scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        image = np.nanmedian(np.stack(diffs), axis=0)
        med_a = np.nanmedian(np.stack([frames_a[i] for i, _, _ in pairs]), axis=0)
        med_b = np.nanmedian(np.stack([frames_b[j] for _, j, _ in pairs]), axis=0)
        reference = np.minimum(
            np.clip(np.nan_to_num(med_a, nan=0.0), 0.0, None),
            np.clip(np.nan_to_num(med_b, nan=0.0), 0.0, None),
        )
    return image, pairs, scales, reference


STAR_RESIDUAL = 0.2      # erwarteter Rest eines stillstehenden Sterns als Anteil seiner Helligkeit


def _mti_significance(image, reference):
    """Signifikanzkarte: D / sqrt(sigma_Rauschen^2 + (0.2 * R_stationary)^2)."""
    valid = np.isfinite(image)
    noise = 1.4826 * float(np.median(np.abs(image[valid] - np.median(image[valid])))) or 1e-9
    ref = np.clip(np.nan_to_num(reference, nan=0.0), 0, None)
    padded = np.pad(ref, 1, mode="edge")
    ny, nx = ref.shape
    local = np.max(np.stack([padded[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)]), axis=0)
    return image / np.sqrt(noise ** 2 + (STAR_RESIDUAL * local) ** 2), noise


def _mti_peak(image, sign, noise, npix):
    """Staerkstes positives (sign=+1) oder negatives (sign=-1) Signal der Signifikanzkarte und seine Lage."""
    work = np.where(np.isfinite(image), sign * image, -np.inf)
    work[:2, :] = work[-2:, :] = -np.inf
    work[:, :2] = work[:, -2:] = -np.inf
    iy, ix = np.unravel_index(np.argmax(work), work.shape)
    if not np.isfinite(work[iy, ix]):
        return None
    y0, y1, x0, x1 = max(iy - 1, 0), iy + 2, max(ix - 1, 0), ix + 2
    stamp = np.clip(np.nan_to_num(sign * image[y0:y1, x0:x1], nan=0.0, neginf=0.0), 0, None)
    yy, xx = np.indices(stamp.shape)
    total = stamp.sum()
    cx = x0 + (stamp * xx).sum() / total if total > 0 else ix
    cy = y0 + (stamp * yy).sum() / total if total > 0 else iy
    centre = (npix - 1) / 2.0
    return {
        "snr": round(float(work[iy, ix] / noise), 1),
        "east_arcsec": round(float((centre - cx) * PIXEL_SCALE_ARCSEC), 1),
        "north_arcsec": round(float((cy - centre) * PIXEL_SCALE_ARCSEC), 1),
    }


class MtiRequest(BaseModel):
    ra: float
    dec: float
    urls_a: list[str]
    urls_b: list[str]
    cutout_pixels: int = 100
    max_dlam: float = 0.03
    subtract_zodi: bool = True
    mode: str = "einfach"
    include_values: bool = False


def _simple_difference(frame_a, frame_b):
    return frame_b - frame_a


@app.post("/api/mti")
def moving_target_indication(req: MtiRequest):
    if req.mode not in ("einfach", "mitteln"):
        raise HTTPException(status_code=400, detail="mode must be 'einfach' or 'mitteln'.")
    if not req.urls_a or not req.urls_b:
        raise HTTPException(status_code=400, detail="A difference image needs exposures from two passes.")
    if len(req.urls_a) > MAX_STACK or len(req.urls_b) > MAX_STACK:
        raise HTTPException(status_code=400, detail=f"At most {MAX_STACK} exposures per pass.")
    if not (16 <= req.cutout_pixels <= 200):
        raise HTTPException(status_code=400, detail="cutout_pixels must be between 16 and 200.")
    if not (0.001 <= req.max_dlam <= 5.0):
        raise HTTPException(status_code=400, detail="The wavelength gap must be between 0.001 and 5 micrometres.")
    for url in req.urls_a + req.urls_b:
        _check_url(url)
    ra, dec = round(req.ra, 7), round(req.dec, 7)

    def load(url):
        try:
            frame, wave = _north_up_frame(url, ra, dec, req.cutout_pixels, req.subtract_zodi)
        except Exception as e:
            print(f"[Hinweis] Aufnahme fuer das Differenzbild uebersprungen: {getattr(e, 'detail', e)!r}")
            return None
        if np.isfinite(frame).sum() < 10:
            return None
        return frame, wave, url

    with ThreadPoolExecutor(max_workers=8) as pool:
        got_a = [r for r in pool.map(load, req.urls_a) if r is not None]
        got_b = [r for r in pool.map(load, req.urls_b) if r is not None]
    if not got_a or not got_b:
        raise HTTPException(status_code=404, detail="In one of the two passes no exposure was usable.")
    too_far = HTTPException(status_code=404, detail="No two exposures with sufficiently similar wavelength found "
                            f"(allowed gap {req.max_dlam} micrometres). Allow a larger gap or choose another infrared range.")
    extra = {}

    if req.mode == "einfach":
        dlam, i, j = min((abs(a[1] - b[1]), i, j) for i, a in enumerate(got_a) for j, b in enumerate(got_b))
        if dlam > req.max_dlam:
            raise too_far
        fa, fb = got_a[i][0], got_b[j][0]
        image = _simple_difference(fa, fb)
        valid = np.isfinite(image)
        if valid.sum() < 10:
            raise HTTPException(status_code=404, detail="No valid pixel in the difference image.")
        centre = float(np.median(image[valid]))
        # Auch im einfachen Modus: Stationäre helle Sterne über min(A, B) in der Signifikanzkarte dämpfen,
        # während die Rohdaten (image / CSV) exakt B - A bleiben.
        a_sub = fa - float(np.nanmedian(fa))
        b_sub = fb - float(np.nanmedian(fb))
        reference = np.minimum(
            np.clip(np.nan_to_num(a_sub, nan=0.0), 0.0, None),
            np.clip(np.nan_to_num(b_sub, nan=0.0), 0.0, None),
        )
        signif, noise = _mti_significance(image - centre, reference)
        dlams, scales = [dlam], [1.0]
        extra = {
            "url_a": got_a[i][2], "url_b": got_b[j][2],
            "wavelength_a_um": round(float(got_a[i][1]), 4), "wavelength_b_um": round(float(got_b[j][1]), 4),
            "diff_median": round(centre, 4),
            "diff_min": round(float(image[valid].min()), 4), "diff_max": round(float(image[valid].max()), 4),
            "unit": "MJy/sr",
        }
        if req.include_values:
            rows = np.flipud(image)
            extra["values"] = [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in rows]
        limit = 12.0
    else:
        sub = lambda got: [g[0] - np.median(g[0][np.isfinite(g[0])]) for g in got]
        result = _mti_combine(sub(got_a), [g[1] for g in got_a], sub(got_b), [g[1] for g in got_b], req.max_dlam)
        if result is None:
            raise too_far
        image, pairs, scales, reference = result
        valid = np.isfinite(image)
        if valid.sum() < 10:
            raise HTTPException(status_code=404, detail="No valid pixel in the difference image.")
        signif, noise = _mti_significance(image, reference)
        dlams = [p[2] for p in pairs]
        limit = 10.0

    norm = np.arcsinh(4.0 * np.clip(np.where(valid, signif, 0.0) / limit, -1, 1)) / np.arcsinh(4.0)
    gray = np.flipud(((norm * 0.5 + 0.5) * 255).astype(np.uint8))
    alpha = np.flipud(np.where(valid, 255, 0).astype(np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(np.dstack([gray, alpha]), mode="LA").save(buffer, format="PNG", optimize=True)

    return {
        "image_b64": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mode": req.mode,
        "n_pairs": len(dlams),
        "n_a": len(got_a), "n_b": len(got_b),
        "median_dlam_um": round(float(np.median(dlams)), 4),
        "max_dlam_um": round(float(max(dlams)), 4),
        "median_scale": round(float(np.median(scales)), 3),
        "noise_mjy_sr": round(noise, 4),
        "appeared": _mti_peak(signif, +1, 1.0, req.cutout_pixels),
        "vanished": _mti_peak(signif, -1, 1.0, req.cutout_pixels),
        "size_px": req.cutout_pixels,
        "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        **extra,
    }


# --- 6d. ENDPUNKT 5: SPEKTRALE KLASSIFIKATION ---
class ClassifyRequest(BaseModel):
    wave_um: list[float]
    flux_mjy: list[float]
    err_mjy: list[float]
    parallax_mas: float | None = None


@app.post("/api/classify")
def classify_endpoint(req: ClassifyRequest):
    """Ordnet ein gemessenes Spektrum (Wellenlaenge, F_nu, Fehler) einer Objektklasse zu (siehe spherex_classify.py)."""
    if not (len(req.wave_um) == len(req.flux_mjy) == len(req.err_mjy)):
        raise HTTPException(status_code=400, detail="wave_um, flux_mjy and err_mjy must have the same length.")
    if len(req.wave_um) > 5000:
        raise HTTPException(status_code=400, detail="At most 5000 data points.")
    try:
        return classify_spectrum(req.wave_um, req.flux_mjy, req.err_mjy, parallax_mas=req.parallax_mas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")


# --- 6e. STATUS ---
@app.get("/api/source")
def image_source():
    """Zeigt, woher die Bilder kommen."""
    return {"image_source": IMAGE_SOURCE, "s3_bucket": S3_BUCKET, "s3_region": S3_REGION,
            "search_mode": SEARCH_MODE, "irsa_paused_s": max(0, round(_irsa_down_until - time.time())),
            "search": "IRSA SIA, Reserve TAP, danach Zwischenspeicher, danach eigene AWS-Suche", "search_cache_dir": SEARCH_CACHE_DIR,
            "cached_searches": len([f for f in os.listdir(SEARCH_CACHE_DIR) if f.endswith(".json")])
            if os.path.isdir(SEARCH_CACHE_DIR) else 0}


# --- 7. START ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)