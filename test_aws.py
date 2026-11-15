"""
Schnelltest: Kommt dieser Rechner an den AWS-Bucket von SPHEREx?

Start:  python test_aws.py
Pakete: pip install s3fs astropy numpy

Prueft drei Dinge nacheinander und meldet jeweils OK oder den Fehler:
  1. Verbindung: die obersten Ordner des Buckets auflisten.
  2. Eine bekannte Datei (QR2, Komet 3I/ATLAS, Detektor 6) oeffnen und ihre Ebenen anzeigen.
  3. Nur einen Ausschnitt von 100 x 100 Pixeln lesen und die Wellenlaenge in der Bildmitte bestimmen.
"""
import time

import numpy as np
import s3fs
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

BUCKET = "nasa-irsa-spherex"
KEY = ("qr2/level2/2025W31_2A/l2b-v20-2025-264/6/"
       "level2_2025W31_2A_0156_1D6_spx_l2b-v20-2025-264.fits")
OPTS = {"anon": True, "client_kwargs": {"region_name": "us-east-1"}, "default_block_size": 1 << 20}

t = time.time()
fs = s3fs.S3FileSystem(**OPTS)
print("1. Ordner im Bucket:", fs.ls(BUCKET)[:10], f"({time.time() - t:.1f} s)")

t = time.time()
with fits.open(f"s3://{BUCKET}/{KEY}", use_fsspec=True, fsspec_kwargs=OPTS, lazy_load_hdus=True) as hdul:
    hdr = hdul["IMAGE"].header
    print("2. Datei offen:", [h.name for h in hdul], hdr["NAXIS1"], "x", hdr["NAXIS2"], f"({time.time() - t:.1f} s)")

    t = time.time()
    cy, cx = hdr["NAXIS2"] // 2, hdr["NAXIS1"] // 2
    cut = np.array(hdul["IMAGE"].section[cy - 50:cy + 50, cx - 50:cx + 50], dtype=np.float32)
    spec = WCS(header=hdr, fobj=hdul, key="W")
    spec.sip = None
    wave, bw = spec.pixel_to_world(cx, cy)
    print(f"3. Ausschnitt {cut.shape}, Median {np.nanmedian(cut):.3f} MJy/sr, "
          f"Wellenlaenge {wave.to_value(u.um):.4f} um, Bandbreite {bw.to_value(u.um):.4f} um "
          f"({time.time() - t:.1f} s)")
print("Alles OK: spherex_api.py kann die Bilder aus AWS lesen.")
