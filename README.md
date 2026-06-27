# URDF Ops

## License

This project is licensed under AGPL-3.0-only. See `LICENSE` and `NOTICE`.

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

## Remote Docker Training

URDF Ops can launch training on a user-managed SSH machine with Docker:

```bash
docker build --target trainer -t urdf-ops:training .
```

Copy or build that image on the remote machine, then select `Remote Docker machine` in the training dialog and provide the SSH host, user, key path, remote output directory, and image name. Managed cloud providers remain disabled until provider submission, log streaming, cancellation, and artifact retrieval are fully wired.

## Keypoint Observations

UrdfOps owns dataset/perception keypoint extraction and validation. Downstream tools consume the stable contract instead of embedding camera-specific logic.

- Current schema: `urdf-ops.keypoint-observations.v1`
- Validate a batch: `POST /keypoint-observations/validate`
- Inspect the schema identifier: `GET /keypoint-observations/schema`

Each frame observation includes `episode_index`, `frame_index`, optional `camera_name`, and one or more keypoints. A keypoint must include `label`, `confidence`, and either `pixel_xy` or `position_xyz_m`; URDF repair consumers use `position_xyz_m` plus `link_name` for link-space calibration.
