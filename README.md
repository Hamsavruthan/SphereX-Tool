# SPHEREx Blink & Moving Target Comparator (SPHEREx Bildvergleich)

An interactive web-based astronomical analysis tool built for the **NASA Space Apps Challenge**. The application queries multi-epoch infrared spectral image data from NASA's **SPHEREx** mission, generates median-subtracted difference maps, and visualizes proper motion and transient objects across sequential observational passes.

## Key Features
- **Real-Time NASA Data Retrieval:** Queries NASA/IPAC IRSA archives and AWS S3 buckets by sky coordinates (ICRS / Galactic) or target name (`spherex_s3_search.py`, `spherex_api.py`).
- **Moving Target Indication (MTI):** Aligns exposures across different passes and wavelength bands ($\Delta\lambda$), performs median background subtraction (`einfach` and `mitteln` modes), and computes difference images with signal-to-noise significance.
- **Sequential Multi-Pass Visualization:** Combines custom FITS cutout rendering with an interactive **Aladin Lite** sky viewer (`index.html`), allowing users to step chronologically through observational passes to track moving stars (e.g., Barnard's Star, Kapteyn's Star) and transient objects.

## Quick Start (Local Setup)

1. **Clone the repository:**

       git clone https://github.com/Hamsavruthan/SphereX-Tool.git
       cd SphereX-Tool

2. **Create and activate a virtual environment:**

       python -m venv .venv

   **Windows (PowerShell):**

       .venv\Scripts\Activate.ps1

   **macOS / Linux:**

       source .venv/bin/activate

3. **Install dependencies:**

       pip install -r requirements.txt

4. **Start the FastAPI backend:**

       uvicorn spherex_api:app --reload

5. **Open the interface:**
   Open `index.html` in your browser (or navigate to `http://127.0.0.1:8000` if served directly by the backend).

## Data Sources & Technologies
- **NASA Data:** NASA/IPAC Infrared Science Archive (IRSA) & SPHEREx Open Data
- **Backend:** Python 3.13, FastAPI, Uvicorn, NumPy, Astropy, Astroquery, Boto3
- **Frontend:** HTML5, CSS3, JavaScript, CDS Aladin Lite

## Super Intelligence (SI / AI) Disclosure
In accordance with the NASA Space Apps Challenge guidelines: Large Language Models / Super Intelligence tools (Claude / Gemini) were used to assist with code structuring, debugging, and project documentation. All astronomical data processing logic, integration, and final validation were created and verified by the team.
