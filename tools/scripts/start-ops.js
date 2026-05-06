#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..', '..');
const webPort = process.env.URDF_OPS_WEB_PORT || '5174';
const apiPort = process.env.URDF_OPS_API_PORT || '8001';
const apiBaseUrl = process.env.URDF_OPS_BACKEND_URL || `http://127.0.0.1:${apiPort}`;
const webBaseUrl = process.env.URDF_OPS_WEB_URL || `http://127.0.0.1:${webPort}`;
const python = existsSync(join(rootDir, '.venv-lerobot', 'bin', 'python3'))
  ? join(rootDir, '.venv-lerobot', 'bin', 'python3')
  : 'python3';

const managedEnv = {
  ...process.env,
  URDF_OPS_WEB_PORT: webPort,
  URDF_OPS_API_PORT: apiPort,
  URDF_OPS_BACKEND_URL: apiBaseUrl,
  URDF_OPS_API_BASE_URL: apiBaseUrl,
  URDF_STUDIO_TRAINING_LEROBOT_PYTHON: python,
};

const children = [];
function spawnManaged(command, args) {
  const child = spawn(command, args, {
    cwd: rootDir,
    env: managedEnv,
    stdio: 'inherit',
    detached: process.platform !== 'win32',
  });
  children.push(child);
  return child;
}
function stop(signal = 'SIGTERM') {
  for (const child of children) {
    if (!child.pid) continue;
    try {
      if (process.platform !== 'win32') process.kill(-child.pid, signal);
      else child.kill(signal);
    } catch {}
  }
}

console.log(`URDF Ops API: ${apiBaseUrl}`);
console.log(`URDF Ops UI:  ${webBaseUrl}`);
spawnManaged(python, ['-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', apiPort]);
spawnManaged(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev']);
process.on('SIGINT', () => { stop('SIGINT'); process.exit(0); });
process.on('SIGTERM', () => { stop('SIGTERM'); process.exit(0); });
