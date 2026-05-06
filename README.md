# URDF Ops

Standalone training operations workspace split from URDF Studio.

## Run

```bash
npm install
npm run dev
```

The frontend defaults to http://127.0.0.1:5174 and proxies `/api` to http://127.0.0.1:8001.

## Backend

```bash
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

Training endpoints are exposed under `/training/*`.
