# rizzup-img-proc

Standalone backend worker for the `rizzup.co.uk` frontend queue defined in `D:\Web\rizzup.co.uk\docs\backend-api-spec.md`.

This service is designed for Windows and splits responsibilities cleanly:

- Node.js + TypeScript handles Netlify Blobs queue polling, locking, retries, status tracking, result writes, and dead-lettering.
- Python handles local image analysis and GPU-backed preview rendering through PyTorch/CUDA.

## What It Processes

The worker consumes immutable queue records written by the frontend repo:

- `upload_photo`
- `analyze_photo_quality`
- `generate_preview`
- `create_checkout_session`

## Local Windows Setup

1. Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install Python packages:

```powershell
python -m pip install -r requirements.txt
```

3. Install Node packages:

```powershell
npm install
```

4. Create `.env` from `.env.example` and fill in:

- `NETLIFY_SITE_ID`
- `NETLIFY_ACCESS_TOKEN`

5. Build and run:

```powershell
npm run build
npm run start
```

The worker entrypoint is compiled to `dist/src/index.js`, so `npm run start` should be run from this repo after a successful build.

## Archive Storage

Archive writes now go through a backend abstraction:

- `RIZZUP_ARCHIVE_BACKEND=local` keeps the current filesystem behavior
- `RIZZUP_ARCHIVE_BACKEND=sftp` uploads archive files to Synology over SFTP

Archive config:

- `RIZZUP_IMAGE_ARCHIVE_ROOT`
  - local mode: absolute or relative filesystem root on the worker
  - SFTP mode: remote Synology root such as `/volume1/web/rizzup.co.uk/image-jobs`
- `RIZZUP_SOURCE_IMAGE_ROOT`
  - optional in local mode
  - defaults to `artifacts/source-images` in SFTP mode so Python still has a local source file to read
- `RIZZUP_SFTP_HOST`
- `RIZZUP_SFTP_PORT`
- `RIZZUP_SFTP_USERNAME`
- `RIZZUP_SFTP_PASSWORD`
- `RIZZUP_SFTP_STRICT_HOST_KEY`
- `RIZZUP_SFTP_HOST_KEY`

When SFTP mode is enabled:

- uploaded source images are staged locally for Python and archived remotely
- previews/finals are still rendered locally on the worker
- `sourcePath`, `previewPath`, and `finalImagePath` in result records become logical archive paths like `2026/04/07/job_id/generated/preview/natural.png`
- Netlify Blobs queue, status, result, lock, dead-letter, and asset flows stay unchanged

## Output Layout

Generated artifacts are written under `artifacts/` by default:

- `artifacts/renders/*` for local render staging when SFTP mode is enabled
- `artifacts/source-images/*` for local source staging when SFTP mode is enabled

Generated preview and final-image binaries are uploaded to the Netlify Blobs assets store and should be served by opaque asset id.

Result/status/lock/dead-letter data is also written to Netlify Blobs stores, not to local disk.

## Tests

Python pipeline tests:

```powershell
python -m unittest discover -s tests_python -p "test_*.py"
```

Node tests after build:

```powershell
npm run build
node --test dist/tests/**/*.test.js
```

## Operational Notes

- `RIZZUP_PYTHON_EXECUTABLE` defaults to `.venv\Scripts\python.exe`
- `RIZZUP_PYTHON_SCRIPT` defaults to `scripts\gpu_pipeline.py`
- The preview pipeline uses CUDA automatically when `torch.cuda.is_available()` is true
- Preview and final generation use a deterministic enhancement-only pipeline built from face detection, framing, lighting correction, skin cleanup, background treatment, preset tuning, and watermark/upscale steps
- The worker loop is simple on purpose so it can be supervised by Windows Task Scheduler or NSSM
