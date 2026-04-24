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
- optional `PUSHER_APP_ID`, `PUSHER_KEY`, `PUSHER_SECRET`, `PUSHER_CLUSTER` if you want the worker to publish realtime status events

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
- generated preview/final Python subprocess output is archived beside each generated image as raw `<generated-image>.stdout.log` and `<generated-image>.stderr.log` files
- Netlify Blobs queue, status, result, lock, dead-letter, and asset flows stay unchanged

## Environment Variables

Required:

- `NETLIFY_SITE_ID`
- `NETLIFY_ACCESS_TOKEN` or `NETLIFY_AUTH_TOKEN`

Blob stores:

- `RIZZUP_QUEUE_STORE` default `rizzup-job-queue`
- `RIZZUP_STATUS_STORE` default `rizzup-job-status`
- `RIZZUP_RESULTS_STORE` default `rizzup-job-results`
- `RIZZUP_ASSETS_STORE` default `rizzup-job-assets`
- `RIZZUP_LOCKS_STORE` default `rizzup-job-locks`
- `RIZZUP_DEAD_LETTER_STORE` default `rizzup-job-dead-letter`

Worker behavior:

- `RIZZUP_POLL_INTERVAL_MS`
- `RIZZUP_MAX_JOBS_PER_POLL`
- `RIZZUP_LOCK_TTL_MS`
- `RIZZUP_MAX_ATTEMPTS`
- `RIZZUP_RETRY_BASE_DELAY_MS`
- `RIZZUP_RETRY_MAX_DELAY_MS`
- `RIZZUP_WORKER_ID`
- `RIZZUP_RESULTS_PUBLIC_BASE_URL`

Realtime status publishing:

- `PUSHER_APP_ID`
- `PUSHER_KEY`
- `PUSHER_SECRET`
- `PUSHER_CLUSTER`

All four Pusher variables must be set for realtime events to be published. If they are omitted, the worker still functions and the frontend falls back to polling.

Python / model config:

- `RIZZUP_PYTHON_EXECUTABLE`
- `RIZZUP_PYTHON_SCRIPT`
- `RIZZUP_FACE_CASCADE_PATH`
- `RIZZUP_EYE_CASCADE_PATH`
- `RIZZUP_PREVIEW_WATERMARK_TEXT`
- `RIZZUP_PREVIEW_WATERMARK_LOGO_PATH`
- `RIZZUP_FIRERED_ENABLED`
- `RIZZUP_FIRERED_MODEL_ID`
- `RIZZUP_FIRERED_PROMPT`
- `RIZZUP_FIRERED_INFERENCE_STEPS`
- `RIZZUP_FIRERED_TRUE_CFG_SCALE`
- `RIZZUP_ANALYSIS_MAX_SIZE`
- `RIZZUP_PREVIEW_MAX_SIZE`
- `RIZZUP_FINAL_DECISION_MAX_SIZE`
- `RIZZUP_FINAL_MIN_WIDTH`
- `RIZZUP_FINAL_MIN_HEIGHT`

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
- Preview and final generation now attempt FireRed image editing first using `Tongyi-MAI/Z-Image-Turbo` with the prompt `Beautify this image`
- FireRed integration is controlled with:
  - `RIZZUP_FIRERED_ENABLED` default `true`
  - `RIZZUP_FIRERED_MODEL_ID` default `Tongyi-MAI/Z-Image-Turbo`
  - `RIZZUP_FIRERED_PROMPT` default `Beautify this image`
  - `RIZZUP_FIRERED_INFERENCE_STEPS` default `30`
  - `RIZZUP_FIRERED_TRUE_CFG_SCALE` default `4`
- If FireRed cannot be loaded or inferred, the worker falls back to the existing deterministic enhancement pipeline so jobs still complete
- The worker loop is simple on purpose so it can be supervised by Windows Task Scheduler or NSSM
