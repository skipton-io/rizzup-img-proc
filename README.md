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

## Source Image Inputs

The current `rizzup.co.uk` frontend queues upload metadata only. For real preview generation, include one of these optional fields in the `upload_photo` payload:

- `sourcePath`
- `sourceUrl`
- `sourceBlobKey`

This worker currently supports `sourcePath` for local Windows processing. If no source file is attached, it still produces a branded placeholder preview and a fallback quality assessment so the queue flow remains operational.

## Output Layout

Generated artifacts are written under `artifacts/` by default:

- `artifacts/previews/*.png`

Result/status/lock/dead-letter data is written to Netlify Blobs stores, not to local disk.

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
- The worker loop is simple on purpose so it can be supervised by Windows Task Scheduler or NSSM
