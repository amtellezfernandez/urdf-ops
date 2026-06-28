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

## Training Compute

URDF Ops is designed as a training operations platform. The training dialog can launch:

- `Local GPU`: train on the machine running the URDF Ops backend.
- `RunPod SSH pod`: paste the RunPod SSH command, use an SSH key, and run directly inside the active pod.
- `Remote Docker machine`: connect to a user-managed SSH host with Docker and the trainer image.

Native cloud provider APIs such as Modal, RunPod API, Macrodata, and AWS are shown as planned until submit, status polling, log streaming, cancellation, and artifact retrieval are all wired.

### RunPod SSH Pod

Create a RunPod pod with an SSH key added in RunPod, then choose `RunPod SSH pod` in the training compute step. Paste the command RunPod gives you, for example:

```bash
ssh 4clkaznp2byq60-64411f1f@ssh.runpod.io
```

URDF Ops will parse the host and user. Set the private key path on the backend machine, usually:

```bash
~/.ssh/runpod_ed25519
```

The direct SSH runner expects a URDF Ops checkout and Python environment inside the pod, by default:

```bash
/workspace/urdf-ops
python3
```

Use `Run preflight` before launch to verify SSH access, the checkout, Python, CUDA visibility, storage, and dataset reachability.

### Remote Docker Machine

URDF Ops can also launch training on a user-managed SSH machine with Docker:

```bash
docker build --target trainer -t urdf-ops:training .
```

Copy or build that image on the remote machine, then select `Remote Docker machine` in the training dialog and provide the SSH host, user, key path, remote output directory, and image name.

## Keypoint Observations

UrdfOps owns dataset/perception keypoint extraction and validation. Downstream tools consume the stable contract instead of embedding camera-specific logic.

- Current schema: `urdf-ops.keypoint-observations.v1`
- Validate a batch: `POST /keypoint-observations/validate`
- Inspect the schema identifier: `GET /keypoint-observations/schema`

Each frame observation includes `episode_index`, `frame_index`, optional `camera_name`, and one or more keypoints. A keypoint must include `label`, `confidence`, and either `pixel_xy` or `position_xyz_m`; URDF repair consumers use `position_xyz_m` plus `link_name` for link-space calibration.
