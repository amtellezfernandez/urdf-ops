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

## Keypoint Observations

UrdfOps owns dataset/perception keypoint extraction and validation. Downstream tools consume the stable contract instead of embedding camera-specific logic.

- Current schema: `urdf-ops.keypoint-observations.v1`
- Validate a batch: `POST /keypoint-observations/validate`
- Inspect the schema identifier: `GET /keypoint-observations/schema`

Each frame observation includes `episode_index`, `frame_index`, optional `camera_name`, and one or more keypoints. A keypoint must include `label`, `confidence`, and either `pixel_xy` or `position_xyz_m`; URDF repair consumers use `position_xyz_m` plus `link_name` for link-space calibration.
