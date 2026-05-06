# URDF Ops

Standalone training operations workspace split from URDF Studio.

## Run Just Ops

```bash
npm install
npm run start
```

The frontend defaults to http://127.0.0.1:5174 and proxies `/api` to http://127.0.0.1:8001.

## Run UI And Backend Separately

```bash
npm run backend
npm run dev
```

Training endpoints are exposed under `/training/*`. URDF Studio can also launch this repo as a sibling checkout during `npm run start`.
