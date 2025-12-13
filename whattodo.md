Control plane (API + UI + registry)

Data plane (artifact store + metrics stream)

Execution plane (runners/agents that actually launch jobs)



UI is for inspection and audit.
There are three common execution modes you should support:

Local runs (developer laptop / workstation)

On-prem GPU cluster (Slurm/Kubernetes)

NVIDIA stack (NGC / Base Command / DGX environment)


git commit / repo ref

container image (or conda env)

command + args

dataset IDs + resolved indices

env vars and secrets references

resource request (gpus, cpu, ram)

tags/metadata

UI Responsibilities

Browse dataset manifests

Visualize dataset composition (pie charts, tables)

Inspect episode provenance

Compare dataset versions

Show experiment → dataset graph




UI is read-only (mostly)

Backend You Actually Need

At minimum:

FastAPI

SQLite / Postgres

Dataset registry table

Experiment registry table

Artifact store (filesystem / S3 / NGC)


Guardrails You Should Implement

Datasets must be frozen before training

Training refuses mutable datasets

Dataset manifests are immutable

Every experiment logs dataset hash

Every artifact is content-addressed



) Run (Experiment)

A Run is an immutable record containing:

config (flattened)

git commit + dirty state

environment fingerprint (CUDA, driver, container digest)

links to datasets (by immutable IDs + hashes)

metrics stream

artifacts index

B) Dataset (Versioned + Composable)

A Dataset is not “a folder.” It is:

a manifest describing sources and selection logic

a resolved index of episode IDs

content hashes

immutable once frozen

Supports dataset-of-datasets and explicit mixes (0.2/0.8) with deterministic resolution.

C) Artifact (Content-addressed)

Artifacts are:

checkpoints, videos, evaluation reports, resolved indices

addressed by hash (CAS-style) and referenced by runs

stored on filesystem/S3/minio/NGC—doesn’t matter

These three primitives let you build everything else without regret.

3. The Minimum Feature Set to Be “W&B/MLflow-Class”

If you want credibility, these features are table stakes:

Logging SDK (Python-first)

init_run(), log_metrics(), log_artifact(), link_dataset()

asynchronous, resilient, works offline

retries and local spool when network is down

Storage + Registry

Run registry (DB)

Dataset registry (DB)

Artifact index + blob store (CAS)

Query

filter runs by tags/config fields/dataset IDs/commit

compare runs

export as parquet/csv

UI

run list + filters

run detail: config, metrics plots, artifacts, videos

dataset detail: composition + provenance graph

diff view: run A vs run B

Governance for Manufacturing

immutability controls

role-based access (even basic at first)

audit log for mutations (dataset creation, tagging, deletion)

retention policies

4. What Makes ROBOT OPS Differentiated (Robotics-Native)

This is where you beat W&B/MLflow:

Episode-first data model

Episode = primary unit

Support “episode lineage” (where it came from, what transforms applied)

Mixed datasets as first-class citizens

explicit composition

deterministic resolution

resolved episode index stored as artifact

“dataset diff” across versions

Rollouts as artifacts with semantic indexing

auto-link videos to episode IDs, policies, environment versions

store action traces alongside video timestamps

Sim/Real alignment tracking

environment fingerprinting (Isaac/Unity/Gym versions, assets hash)

calibration versioning for real robots

safety events and anomalies