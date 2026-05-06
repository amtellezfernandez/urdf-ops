#!/usr/bin/env node

import { appendFileSync, readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { execFileSync, execSync, spawnSync } from 'child_process';
import readline from 'readline';
import { maskToken, resolveSetupGitHubToken } from './githubAuth.js';
import {
  buildSetupSummarySections,
  isTruthyEnvValue,
  selectInstalledStalePythonDependencies,
  shouldInstallGlobalIlu,
} from './setupHelpers.js';
import { OPENARM_HARDWARE_PIP_DEPENDENCIES } from './openArmHardwareParams.js';
import { buildOpenArmHardwareVerifyImportScript } from './openArmHardwareRuntime.js';
import {
  BACKEND_PYTHON_DEPENDENCIES,
  BACKEND_PYTHON_STALE_DEPENDENCIES,
  BACKEND_PYTHON_VERIFY_IMPORT_SCRIPT,
  GITHUB_CLI_LOGIN_COMMAND,
  GITHUB_FINE_GRAINED_TOKEN_URL,
  GLOBAL_ILU_INSTALL_COMMAND,
  HUGGING_FACE_TOKEN_URL,
  LEROBOT_TOOLCHAIN_DIRNAME,
  LEROBOT_TRAINING_DEPENDENCIES,
  LEROBOT_TRAINING_VERIFY_IMPORT_SCRIPT,
  LOCAL_ILU_COMMAND,
  MJLAB_DEPENDENCIES,
  MJLAB_SKIP_AUTO_INSTALL_ENV,
  MJLAB_VERIFY_IMPORT_SCRIPT,
  PYTHON_ENV_DIRNAME,
} from './setupParams.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..', '..');

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  pink: '\x1b[35m',      // Magenta/pink
  pinkBright: '\x1b[95m', // Bright magenta
  pinkLight: '\x1b[38;5;213m', // Light pink
  pinkDark: '\x1b[38;5;162m',  // Dark pink
  purple: '\x1b[38;5;129m',    // Purple
  purpleBright: '\x1b[38;5;141m', // Bright purple
  purpleLight: '\x1b[38;5;183m',   // Light purple
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  gray: '\x1b[90m',
  underline: '\x1b[4m',
};

const banner = `
${colors.pinkBright}    __  ______  ____  ______   _____ __            ___    ${colors.reset}
${colors.pinkBright}   / / / / __ \\/ __ \\/ ____/  / ___// /___  ______/ (_)___ ${colors.reset}
${colors.pink}  / / / / /_/ / / / / /_      \\__ \\/ __/ / / / __  / / __ \\${colors.reset}
${colors.pink} / /_/ / _, _/ /_/ / __/     ___/ / /_/ /_/ / /_/ / / /_/ /${colors.reset}
${colors.pinkLight} \\____/_/ |_/_____/_/       /____/\\__/\\__,_/\\__,_/_/\\____/ ${colors.reset}
${colors.reset}                                                            

${colors.gray}─────────────────────────────────────────────────────────────${colors.reset}
`;

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function logArrow(message) {
  log(`→ ${message}`, colors.pink);
}

function logSuccess(message) {
  log(`✓ ${message}`, colors.green);
}

function logInfo(message) {
  log(`  ${message}`, colors.gray);
}

function logUrl(url, text) {
  const underline = '\x1b[4m';
  log(`  ${text}: ${colors.pinkBright}${underline}${url}${colors.reset}`);
}

function isInteractive() {
  return Boolean(process.stdin.isTTY && process.stdout.isTTY);
}

function getUvEnv() {
  const uvCacheDir = process.env.UV_CACHE_DIR || join(rootDir, '.uv-cache');
  return { ...process.env, UV_CACHE_DIR: uvCacheDir };
}

function question(query) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) => {
    rl.question(query, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

function getNpmCommand() {
  const npmExecPath = typeof process.env.npm_execpath === 'string' ? process.env.npm_execpath.trim() : '';
  if (npmExecPath) {
    return {
      command: process.execPath,
      argsPrefix: [npmExecPath],
    };
  }

  return {
    command: process.platform === 'win32' ? 'npm.cmd' : 'npm',
    argsPrefix: [],
  };
}

function runNpm(args, options = {}) {
  const { command, argsPrefix } = getNpmCommand();
  execFileSync(command, [...argsPrefix, ...args], {
    cwd: rootDir,
    stdio: 'inherit',
    ...options,
  });
}

async function installDependencies() {
  logArrow('Installing dependencies...');
  
  try {
    const nodeModulesPath = join(rootDir, 'node_modules');
    const viteBin = join(nodeModulesPath, '.bin', 'vite');
    if (!existsSync(nodeModulesPath) || !existsSync(viteBin)) {
      runNpm(['install']);
    } else {
      const inquirerPath = join(rootDir, 'node_modules', 'inquirer');
      if (!existsSync(inquirerPath)) {
        logInfo('Installing inquirer...');
        runNpm(['install', 'inquirer']);
      }
    }
    logSuccess('Dependencies installed successfully');
    logInfo(`Local i-love-urdf CLI is available in this repo via: ${colors.pinkBright}${LOCAL_ILU_COMMAND}${colors.reset}`);
  } catch (error) {
    log('✗ Failed to install dependencies', colors.yellow);
    throw error;
  }
}

function getConfigPath() {
  return join(rootDir, '.urdf-studio-config.json');
}

function getAppConfigPath() {
  return join(rootDir, 'config', 'app.config.json');
}

function loadAppConfig() {
  const appConfigPath = getAppConfigPath();
  if (!existsSync(appConfigPath)) {
    return {};
  }
  try {
    return JSON.parse(readFileSync(appConfigPath, 'utf-8'));
  } catch (e) {
    return {};
  }
}

function loadConfig() {
  const configPath = getConfigPath();
  if (existsSync(configPath)) {
    try {
      const content = readFileSync(configPath, 'utf-8');
      return JSON.parse(content);
    } catch (e) {
      return {};
    }
  }
  return {};
}

function saveConfig(config) {
  const configPath = getConfigPath();
  writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
}

async function setupHuggingFace() {
  log('');
  logArrow('🤗 HuggingFace Authentication (Optional)');
  log('');
  
  const config = loadConfig();
  const currentToken = config.huggingfaceToken || '';
  const configPath = getConfigPath();

  if (process.env.URDF_STUDIO_SKIP_TOKENS) {
    logInfo('Token setup skipped (URDF_STUDIO_SKIP_TOKENS is set).');
    return;
  }

  if (!isInteractive()) {
    logInfo('Non-interactive session detected. Skipping HuggingFace token setup.');
    return;
  }

  const hfAnswer = (await question('  Set up HuggingFace token now? (y/N): ')).trim().toLowerCase();
  if (hfAnswer !== 'y' && hfAnswer !== 'yes') {
    logInfo('HuggingFace token setup skipped by user.');
    return;
  }

  if (currentToken) {
    const maskedToken = maskToken(currentToken);
    logInfo(`Current token: ${colors.pinkBright}${maskedToken}${colors.reset}`);
    log('');
    logInfo('Options:');
    logInfo('  [r] Remove token');
    logInfo('  [s] Substitute/Update token');
    logInfo('  [Enter] Skip (keep current)');
    log('');
    
    const action = (await question(`  Choose an option: ${colors.pinkBright}`)).trim().toLowerCase();
    
    if (action === 'r' || action === 'remove') {
      delete config.huggingfaceToken;
      saveConfig(config);
      logSuccess('HuggingFace token removed');
      return;
    }
    if (action !== 's' && action !== 'substitute' && action !== 'update') {
      logInfo('Token unchanged (keeping current token).');
      return;
    }
  } else {
    logInfo('A token is required for uploading and managing datasets.');
    log('');
    logInfo('To create a token:');
    logUrl(HUGGING_FACE_TOKEN_URL, 'Visit');
    logInfo('1. Click "New token"');
    logInfo('2. Set permissions: Read access to repos + Write access to repos');
    logInfo('3. Copy the token (starts with hf_)');
    log('');
  }
  
  logInfo(`${colors.yellow}⚠ Security: Your token will be saved locally on your computer, keep it private and never share it.${colors.reset}`);
  logInfo(`   Saved to: ${colors.gray}${configPath}${colors.reset}`);
  log('');
  
  // Ask for token input (completely hidden)
  let token = '';
  try {
    const inquirer = (await import('inquirer')).default;
    ({ token } = await inquirer.prompt([
      {
        type: 'password',
        name: 'token',
        message: `${colors.pinkBright}  Enter your HuggingFace token (or press Enter to skip):${colors.reset}`,
        mask: '', // Completely hidden
      },
    ]));
  } catch (e) {
    logInfo(`Token prompt unavailable (${e?.message || 'unknown error'}).`);
    token = (await question('  Enter your HuggingFace token (visible input, or press Enter to skip): ')).trim();
  }
  
  if (token?.trim()) {
    config.huggingfaceToken = token.trim();
    saveConfig(config);
    logSuccess('HuggingFace token saved');
    logInfo(`   Location: ${colors.gray}${configPath}${colors.reset}`);
  } else {
    logInfo('No token entered. Token setup cancelled.');
  }
}

async function setupGitHub() {
  log('');
  logArrow('🐙 GitHub Access (Optional)');
  log('');
  
  const config = loadConfig();
  const currentToken = config.githubToken || '';
  const configPath = getConfigPath();

  if (process.env.URDF_STUDIO_SKIP_TOKENS) {
    logInfo('Token setup skipped (URDF_STUDIO_SKIP_TOKENS is set).');
    return;
  }

  if (!isInteractive()) {
    logInfo('Non-interactive session detected. Skipping GitHub token setup.');
    return;
  }

  const ghAnswer = (await question('  Configure GitHub access now? (y/N): ')).trim().toLowerCase();
  if (ghAnswer !== 'y' && ghAnswer !== 'yes') {
    logInfo('GitHub access setup skipped by user.');
    return;
  }

  if (currentToken) {
    const maskedToken = maskToken(currentToken);
    logInfo(`Current token: ${colors.purpleBright}${maskedToken}${colors.reset}`);
    log('');
    logInfo('Options:');
    logInfo('  [r] Remove token');
    logInfo('  [s] Substitute/Update token');
    logInfo('  [Enter] Skip (keep current)');
    log('');
    
    const action = (await question(`  Choose an option: ${colors.purpleBright}`)).trim().toLowerCase();
    
    if (action === 'r' || action === 'remove') {
      delete config.githubToken;
      saveConfig(config);
      logSuccess('GitHub token removed');
      return;
    }
    if (action !== 's' && action !== 'substitute' && action !== 'update') {
      logInfo('Token unchanged (keeping current token).');
      return;
    }
  }

  const detectedGitHubAuth = resolveSetupGitHubToken();
  if (!currentToken && detectedGitHubAuth.token) {
    const maskedDetectedToken = maskToken(detectedGitHubAuth.token);
    logInfo(
      `Detected GitHub access via ${colors.purpleBright}${detectedGitHubAuth.source}${colors.reset}: ${colors.purpleBright}${maskedDetectedToken}${colors.reset}`
    );
    logInfo('URDF Studio can already reuse this access without saving a local token.');
    log('');
    logInfo('Options:');
    logInfo('  [Enter] Keep using detected access (recommended)');
    logInfo('  [s] Save detected token locally');
    logInfo('  [m] Enter a different token manually');
    log('');

    const detectedAction = (await question(`  Choose an option: ${colors.purpleBright}`)).trim().toLowerCase();
    if (detectedAction === '' || detectedAction === 'k' || detectedAction === 'keep') {
      logInfo('Detected GitHub access will be reused without saving a local token.');
      return;
    }
    if (detectedAction === 's' || detectedAction === 'save') {
      config.githubToken = detectedGitHubAuth.token;
      saveConfig(config);
      logSuccess('GitHub token saved');
      logInfo(`   Source: ${colors.gray}${detectedGitHubAuth.source}${colors.reset}`);
      logInfo(`   Location: ${colors.gray}${configPath}${colors.reset}`);
      return;
    }
    if (detectedAction !== 'm' && detectedAction !== 'manual') {
      logInfo('Detected GitHub access not saved. You can still enter a token manually later.');
      return;
    }
    logInfo('Detected GitHub access not saved. Enter a different token below if you still want a local fallback.');
    log('');
  }

  if (!currentToken) {
    logInfo('Recommended GitHub access options:');
    logInfo(`1. Run ${colors.purpleBright}${GITHUB_CLI_LOGIN_COMMAND}${colors.reset} (recommended, nothing stored locally)`);
    logInfo('2. Export GITHUB_TOKEN or GH_TOKEN in your shell');
    logInfo('3. Save a fine-grained token locally for URDF Studio only');
    log('');
    logInfo('If you want to create a token:');
    logUrl(GITHUB_FINE_GRAINED_TOKEN_URL, 'Visit');
    logInfo('1. Click "Generate new token (Fine-grained)"');
    logInfo('2. Under Repository access, choose:');
    logInfo('   ✓ Only select repositories');
    logInfo('   (Pick the repos you want URDF Studio to access)');
    logInfo('3. Under Repository permissions, enable:');
    logInfo('   Contents → Read and write');
    logInfo('   Pull requests → Read and write');
    logInfo('   Metadata → Read (usually enabled by default)');
    logInfo('4. Generate the token and copy it (it will look like github_pat_...)');
    log('');
  }

  logInfo(`${colors.yellow}⚠ Security: Your token is stored locally on your computer only.${colors.reset}`);
  logInfo(`   It is never shared or uploaded anywhere.${colors.reset}`);
  logInfo(`   Saved to: ${colors.gray}${configPath}${colors.reset}`);
  log('');
  
  // Ask for token input (completely hidden)
  let token = '';
  try {
    const inquirer = (await import('inquirer')).default;
    ({ token } = await inquirer.prompt([
      {
        type: 'password',
        name: 'token',
        message: `${colors.purpleBright}  Enter your GitHub token (or press Enter to skip):${colors.reset}`,
        mask: '', // Completely hidden
      },
    ]));
  } catch (e) {
    logInfo(`Token prompt unavailable (${e?.message || 'unknown error'}).`);
    token = (await question('  Enter your GitHub token (visible input, or press Enter to skip): ')).trim();
  }
  
  if (token?.trim()) {
    config.githubToken = token.trim();
    saveConfig(config);
    logSuccess('GitHub token saved');
    logInfo(`   Location: ${colors.gray}${configPath}${colors.reset}`);
  } else {
    logInfo('No token entered. Token setup cancelled.');
  }
}

async function installOptionalGlobalIlu() {
  if (!shouldInstallGlobalIlu()) {
    return {
      attempted: false,
      installed: false,
    };
  }

  const localIluPackagePath = join(rootDir, 'node_modules', 'i-love-urdf');
  if (!existsSync(localIluPackagePath)) {
    log('✗ Optional global ilu install requested, but i-love-urdf is not installed locally.', colors.yellow);
    logInfo(`Local CLI still works via ${LOCAL_ILU_COMMAND}`);
    return {
      attempted: true,
      installed: false,
    };
  }

  log('');
  logArrow('🧰 Installing optional global i-love-urdf CLI');
  log('');

  try {
    runNpm(['install', '-g', localIluPackagePath]);
    logSuccess('Global ilu CLI installed');
    return {
      attempted: true,
      installed: true,
    };
  } catch (_error) {
    log('✗ Failed to install the optional global ilu CLI', colors.yellow);
    logInfo(`Retry later with: ${GLOBAL_ILU_INSTALL_COMMAND}`);
    logInfo(`Local CLI still works via ${LOCAL_ILU_COMMAND}`);
    return {
      attempted: true,
      installed: false,
    };
  }
}

function printSetupSummary({ globalIluResult } = {}) {
  log('');
  logArrow('Setup summary');
  log('');

  const sections = buildSetupSummarySections({
    globalIluAttempted: Boolean(globalIluResult?.attempted),
    globalIluInstalled: Boolean(globalIluResult?.installed),
  });

  for (const section of sections) {
    log(section.heading, colors.bright);
    for (const line of section.lines) {
      logInfo(line);
    }
    log('');
  }
}

function findUv() {
  // Check common installation locations for uv
  const uvLocations = [
    join(process.env.HOME || '', '.local', 'bin', 'uv'),
    join(process.env.HOME || '', '.cargo', 'bin', 'uv'),
    '/usr/local/bin/uv',
    '/usr/bin/uv',
  ];

  for (const uvPath of uvLocations) {
    if (existsSync(uvPath)) {
      return uvPath;
    }
  }

  // Try to find uv in PATH
  const pathEnv = process.env.PATH || '';
  for (const dir of pathEnv.split(':')) {
    if (!dir) {
      continue;
    }
    const candidate = join(dir, 'uv');
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

function findCargo() {
  const cargoLocations = [
    join(process.env.HOME || '', '.cargo', 'bin', 'cargo'),
    '/usr/local/bin/cargo',
    '/usr/bin/cargo',
  ];

  for (const cargoPath of cargoLocations) {
    if (existsSync(cargoPath)) {
      return cargoPath;
    }
  }

  const pathEnv = process.env.PATH || '';
  for (const dir of pathEnv.split(':')) {
    if (!dir) {
      continue;
    }
    const candidate = join(dir, 'cargo');
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

function ensureCargoPathInShellRc() {
  const home = process.env.HOME || '';
  if (!home) {
    return;
  }

  const bashRc = join(home, '.bashrc');
  const exportLine = 'export PATH="$HOME/.cargo/bin:$PATH"';
  const marker = '# Added by URDF Studio setup: Rust cargo bin';

  let needsAppend = true;
  if (existsSync(bashRc)) {
    try {
      const content = readFileSync(bashRc, 'utf-8');
      if (content.includes(exportLine)) {
        needsAppend = false;
      }
    } catch (e) {
      needsAppend = true;
    }
  }

  if (needsAppend) {
    try {
      appendFileSync(bashRc, `\n${marker}\n${exportLine}\n`, 'utf-8');
      logSuccess('Added Rust cargo path to ~/.bashrc');
    } catch (e) {
      logInfo('Could not update ~/.bashrc automatically. You may need to add cargo path manually.');
    }
  }

  // Make cargo visible to subsequent setup steps in this process.
  const cargoBin = join(home, '.cargo', 'bin');
  if (!String(process.env.PATH || '').split(':').includes(cargoBin)) {
    process.env.PATH = `${cargoBin}:${process.env.PATH || ''}`;
  }
}

function shouldAutoInstallRust() {
  if (/^(1|true|yes)$/i.test(process.env.URDF_STUDIO_SKIP_RUST_AUTO_INSTALL || '')) {
    return false;
  }
  if (/^(0|false|no)$/i.test(process.env.URDF_STUDIO_AUTO_INSTALL_RUST || '')) {
    return false;
  }
  if (/^(1|true|yes)$/i.test(process.env.URDF_STUDIO_AUTO_INSTALL_RUST || '')) {
    return true;
  }
  // Default to auto-install so ikd setup is turnkey when enabled.
  return true;
}

function installRustToolchain() {
  logInfo('Installing Rust toolchain with rustup (minimal profile)...');
  execSync(
    'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal',
    {
      cwd: rootDir,
      stdio: 'inherit',
      shell: true,
    }
  );
}

async function checkIkd() {
  log('');
  logArrow('🦀 Checking native IKD toolchain');
  log('');

  const appConfig = loadAppConfig();
  const ikdConfig = appConfig?.ikd || {};
  const ikdEnabled = Boolean(ikdConfig.enabled);
  const ikdManifest = join(rootDir, 'ikd', 'Cargo.toml');
  const ikdPresent = existsSync(ikdManifest);
  let cargoPath = findCargo();

  if (!ikdEnabled && !ikdPresent) {
    logInfo('ikd is not enabled and native daemon files were not found.');
    return true;
  }

  if (!ikdEnabled && ikdPresent) {
    logInfo('ikd is present in this repo. Installing Rust prerequisites automatically.');
  }

  if (!cargoPath) {
    log('✗ ikd is enabled, but cargo was not found.', colors.yellow);
    let shouldInstall = shouldAutoInstallRust();

    if (shouldInstall) {
      try {
        installRustToolchain();
        ensureCargoPathInShellRc();
        cargoPath = findCargo();
      } catch (e) {
        log('✗ Rust auto-install failed.', colors.yellow);
      }
    }

    if (!cargoPath) {
      logInfo('Install Rust toolchain manually:');
      logInfo('  curl --proto \"=https\" --tlsv1.2 -sSf https://sh.rustup.rs | sh');
      logInfo('Then restart your shell and run setup again.');
      logInfo('Auto-install is enabled by default; disable with URDF_STUDIO_SKIP_RUST_AUTO_INSTALL=1');
      logInfo('Or disable ikd with: config/app.config.json -> ikd.enabled=false');
      return false;
    }
  }

  logSuccess(`Found cargo at: ${cargoPath}`);
  ensureCargoPathInShellRc();
  try {
    execFileSync(cargoPath, ['--version'], { stdio: 'inherit' });
  } catch (e) {
    log('✗ cargo exists but failed to run.', colors.yellow);
    logInfo('Reinstall Rust toolchain or disable ikd in config/app.config.json.');
    return false;
  }
  if (!existsSync(ikdManifest)) {
    log('✗ ikd is enabled but ikd/Cargo.toml is missing.', colors.yellow);
    logInfo('Check your branch or set ikd.enabled=false.');
    return false;
  }

  logSuccess('ikd toolchain prerequisites look good');
  return true;
}

function getManagedPythonPath() {
  return join(rootDir, PYTHON_ENV_DIRNAME, 'bin', 'python3');
}

async function setupPythonBackendEnvironment() {
  log('');
  logArrow('🔍 Setting up unified Python backend/training environment');
  log('');

  const venvPath = join(rootDir, PYTHON_ENV_DIRNAME);
  const venvPython = getManagedPythonPath();

  const uvPath = findUv();
  if (!uvPath) {
    log('✗ uv not found. Please install uv first:', colors.yellow);
    log('');
    logInfo('Install uv with:');
    logInfo('  curl -LsSf https://astral.sh/uv/install.sh | sh');
    log('');
    return false;
  }

  logSuccess(`Found uv at: ${uvPath}`);

  if (existsSync(venvPython)) {
    logInfo(`Unified Python environment already exists at ${PYTHON_ENV_DIRNAME}`);
    return true;
  }

  const pythonPath = findPythonForLeRobot();
  if (!pythonPath) {
    log('✗ Python 3.12+ was not found for the unified Python environment.', colors.yellow);
    logInfo('Install Python 3.12+ or set URDF_STUDIO_LEROBOT_BOOTSTRAP_PYTHON=/path/to/python3.12');
    return false;
  }

  logInfo(`Creating ${venvPath} with ${pythonPath}`);
  try {
    execFileSync(uvPath, ['venv', '--python', pythonPath, venvPath], {
      cwd: rootDir,
      stdio: 'inherit',
      env: getUvEnv(),
    });
    logSuccess('Unified Python environment created');
    return true;
  } catch (e) {
    log('✗ Failed to create unified Python environment', colors.yellow);
    return false;
  }
}

function findPythonForLeRobot() {
  const configuredPython = process.env.URDF_STUDIO_LEROBOT_BOOTSTRAP_PYTHON;
  const candidates = [
    configuredPython,
    'python3.13',
    'python3.12',
    '/usr/bin/python3.13',
    '/usr/bin/python3.12',
    join(process.env.HOME || '', 'miniconda3', 'bin', 'python3.13'),
    join(process.env.HOME || '', 'miniconda3', 'bin', 'python3.12'),
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      const version = execFileSync(
        candidate,
        ['-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
        { encoding: 'utf-8' }
      ).trim();
      const [major, minor] = version.split('.').map(Number);
      if (major > 3 || (major === 3 && minor >= 12)) {
        return candidate;
      }
    } catch (e) {
      // Try the next candidate.
    }
  }

  return null;
}

function shouldInstallOfficialLeRobot() {
  return !isTruthyEnvValue(process.env.URDF_STUDIO_SKIP_LEROBOT_AUTO_INSTALL);
}

function shouldInstallMjlab() {
  return !isTruthyEnvValue(process.env[MJLAB_SKIP_AUTO_INSTALL_ENV]);
}

function listInstalledPythonPackageNames(venvPython) {
  const script = [
    'import importlib.metadata as metadata',
    'import json',
    'names = []',
    'for distribution in metadata.distributions():',
    '    name = distribution.metadata.get("Name")',
    '    if name:',
    '        names.append(name)',
    'print(json.dumps(names))',
  ].join('\n');

  const result = spawnSync(venvPython, ['-c', script], {
    cwd: rootDir,
    encoding: 'utf-8',
  });

  if (result.status !== 0 || !result.stdout) {
    const stderr = String(result.stderr || '').trim();
    throw new Error(stderr || 'Installed Python package inspection failed');
  }

  const output = result.stdout;
  return JSON.parse(output);
}

async function installOfficialLeRobotToolchain() {
  log('');
  logArrow('🤖 Installing official LeRobot training toolchain');
  log('');

  if (!shouldInstallOfficialLeRobot()) {
    logInfo('Skipping official LeRobot install because URDF_STUDIO_SKIP_LEROBOT_AUTO_INSTALL is set.');
    return true;
  }

  const uvPath = findUv();
  if (!uvPath) {
    log('✗ uv not found. Official LeRobot dataset merge requires uv.', colors.yellow);
    return false;
  }

  const toolchainPath = join(rootDir, LEROBOT_TOOLCHAIN_DIRNAME);
  const toolchainPython = join(toolchainPath, 'bin', 'python3');
  if (!existsSync(toolchainPython)) {
    const pythonPath = findPythonForLeRobot();
    if (!pythonPath) {
      log('✗ Python 3.12+ was not found for official LeRobot.', colors.yellow);
      logInfo('Install Python 3.12+ or set URDF_STUDIO_LEROBOT_BOOTSTRAP_PYTHON=/path/to/python3.12');
      return false;
    }
    logInfo(`Creating ${toolchainPath} with ${pythonPath}`);
    execFileSync(uvPath, ['venv', '--python', pythonPath, toolchainPath], {
      cwd: rootDir,
      stdio: 'inherit',
      env: getUvEnv(),
    });
  } else {
    logInfo('Unified Python environment already exists');
  }

  try {
    execFileSync(
      toolchainPython,
      ['-c', LEROBOT_TRAINING_VERIFY_IMPORT_SCRIPT],
      { cwd: rootDir, stdio: 'inherit' }
    );
    logSuccess('Official LeRobot training runtime already installed');
    return true;
  } catch (e) {
    logInfo(`Installing official LeRobot training packages in ${LEROBOT_TOOLCHAIN_DIRNAME}...`);
  }

  try {
    execFileSync(uvPath, ['pip', 'install', '--python', toolchainPython, ...LEROBOT_TRAINING_DEPENDENCIES], {
      cwd: rootDir,
      stdio: 'inherit',
      env: getUvEnv(),
    });
    execFileSync(
      toolchainPython,
      ['-c', LEROBOT_TRAINING_VERIFY_IMPORT_SCRIPT],
      { cwd: rootDir, stdio: 'inherit' }
    );
    logSuccess('Official LeRobot training runtime installed');
    logInfo(`Backend will use unified Python: ${toolchainPython}`);
    return true;
  } catch (e) {
    log('✗ Failed to install official LeRobot training runtime', colors.yellow);
    logInfo('Try manually:');
    logInfo(`  "${uvPath}" pip install --python ${LEROBOT_TOOLCHAIN_DIRNAME}/bin/python3 ${LEROBOT_TRAINING_DEPENDENCIES.join(' ')}`);
    return false;
  }
}

async function installOpenArmHardwareRuntime() {
  log('');
  logArrow('🦾 Installing OpenArm hardware runtime');
  log('');

  const venvPython = getManagedPythonPath();
  const uvPath = findUv();
  if (!existsSync(venvPython)) {
    logInfo(`Unified Python environment not found at ${venvPython}. Run setup first.`);
    return false;
  }
  if (!uvPath) {
    log('✗ uv not found. OpenArm hardware setup requires uv.', colors.yellow);
    return false;
  }

  const verifyScript = buildOpenArmHardwareVerifyImportScript();
  try {
    execFileSync(venvPython, ['-c', verifyScript], {
      cwd: rootDir,
      stdio: 'inherit',
    });
    logSuccess('OpenArm hardware runtime already installed');
    return true;
  } catch (e) {
    logInfo(`Installing OpenArm hardware packages in ${PYTHON_ENV_DIRNAME}...`);
  }

  try {
    execFileSync(
      uvPath,
      ['pip', 'install', '--python', venvPython, ...OPENARM_HARDWARE_PIP_DEPENDENCIES],
      {
        cwd: rootDir,
        stdio: 'inherit',
        env: getUvEnv(),
      }
    );
    execFileSync(venvPython, ['-c', verifyScript], {
      cwd: rootDir,
      stdio: 'inherit',
    });
    logSuccess('OpenArm hardware runtime installed');
    return true;
  } catch (e) {
    log('✗ Failed to install OpenArm hardware runtime', colors.yellow);
    logInfo('Try manually:');
    const manualDependencies = OPENARM_HARDWARE_PIP_DEPENDENCIES
      .map((dependency) => JSON.stringify(dependency))
      .join(' ');
    logInfo(`  "${uvPath}" pip install --python ${PYTHON_ENV_DIRNAME}/bin/python3 ${manualDependencies}`);
    return false;
  }
}

async function installMjlabRuntime() {
  log('');
  logArrow('🧪 Installing MJLab validation runtime');
  log('');

  if (!shouldInstallMjlab()) {
    logInfo(`Skipping MJLab install because ${MJLAB_SKIP_AUTO_INSTALL_ENV} is set.`);
    return true;
  }

  const venvPython = getManagedPythonPath();
  const uvPath = findUv();
  if (!existsSync(venvPython)) {
    logInfo(`Unified Python environment not found at ${venvPython}. Run setup first.`);
    return false;
  }
  if (!uvPath) {
    log('✗ uv not found. MJLab setup requires uv.', colors.yellow);
    return false;
  }

  try {
    execFileSync(venvPython, ['-c', MJLAB_VERIFY_IMPORT_SCRIPT], {
      cwd: rootDir,
      stdio: 'inherit',
    });
    logSuccess('MJLab runtime already installed');
    return true;
  } catch (e) {
    logInfo(`Installing MJLab packages in ${PYTHON_ENV_DIRNAME}...`);
  }

  try {
    execFileSync(uvPath, ['pip', 'install', '--python', venvPython, ...MJLAB_DEPENDENCIES], {
      cwd: rootDir,
      stdio: 'inherit',
      env: getUvEnv(),
    });
    execFileSync(venvPython, ['-c', MJLAB_VERIFY_IMPORT_SCRIPT], {
      cwd: rootDir,
      stdio: 'inherit',
    });
    logSuccess('MJLab validation runtime installed');
    return true;
  } catch (e) {
    log('✗ Failed to install MJLab validation runtime', colors.yellow);
    logInfo('Try manually:');
    logInfo(`  "${uvPath}" pip install --python ${PYTHON_ENV_DIRNAME}/bin/python3 ${MJLAB_DEPENDENCIES.map((dependency) => JSON.stringify(dependency)).join(' ')}`);
    return false;
  }
}

async function installBackendDeps() {
  log('');
  logArrow('🔧 Installing backend Python dependencies');
  log('');

  const venvPython = getManagedPythonPath();
  const uvPath = findUv();

  if (!existsSync(venvPython)) {
    logInfo(`Unified Python environment not found at ${venvPython}. Run setup first.`);
    return false;
  }
  if (!uvPath) {
    log('✗ uv not found. Please install uv first:', colors.yellow);
    return false;
  }

  try {
    const installedPackageNames = listInstalledPythonPackageNames(venvPython);
    const installedStaleDependencies = selectInstalledStalePythonDependencies({
      staleDependencies: BACKEND_PYTHON_STALE_DEPENDENCIES,
      installedPackageNames,
    });

    if (installedStaleDependencies.length > 0) {
      logInfo(`Removing stale backend packages: ${installedStaleDependencies.join(', ')}`);
      execFileSync(
        uvPath,
        ['pip', 'uninstall', '--python', venvPython, ...installedStaleDependencies],
        {
          cwd: rootDir,
          stdio: 'inherit',
          env: getUvEnv(),
        }
      );
    }
  } catch (e) {
    logInfo('Continuing after stale backend package cleanup could not inspect or remove obsolete packages.');
  }

  try {
    execFileSync(venvPython, ['-c', BACKEND_PYTHON_VERIFY_IMPORT_SCRIPT], {
      cwd: rootDir,
      stdio: 'inherit',
    });
    logSuccess('Backend Python runtime already installed');
    return true;
  } catch (e) {
    logInfo('Installing or repairing backend Python packages...');
    logInfo(`Installing: ${BACKEND_PYTHON_DEPENDENCIES.join(', ')}`);
  }

  try {
    execFileSync(uvPath, ['pip', 'install', '--python', venvPython, ...BACKEND_PYTHON_DEPENDENCIES], {
      cwd: rootDir,
      stdio: 'inherit',
      env: getUvEnv()
    });
    logInfo('Verifying backend Python runtime...');
    execFileSync(venvPython, ['-c', BACKEND_PYTHON_VERIFY_IMPORT_SCRIPT], {
      cwd: rootDir,
      stdio: 'inherit'
    });
    logSuccess('Backend dependencies installed');
    return true;
  } catch (e) {
    log('✗ Failed to install backend dependencies', colors.yellow);
    logInfo(`   You can try installing manually:`);
    logInfo(`     "${uvPath}" pip install --python ${PYTHON_ENV_DIRNAME}/bin/python3 ${BACKEND_PYTHON_DEPENDENCIES.map((dependency) => JSON.stringify(dependency)).join(' ')}`);
    return false;
  }
}

async function installTwinDepsIfRequested() {
  const shouldInstallTwin =
    process.argv.includes('--twin') ||
    process.argv.includes('--install-twin') ||
    process.env.npm_config_twin === 'true' ||
    process.env.npm_config_twin === '1' ||
    process.env.TWIN === 'true' ||
    process.env.TWIN === '1';

  if (!shouldInstallTwin) {
    return;
  }

  const twinScript = join(rootDir, 'scripts', 'twin.js');
  if (!existsSync(twinScript)) {
    log('✗ Twin setup requested but scripts/twin.js was not found', colors.yellow);
    throw new Error('Missing scripts/twin.js');
  }

  log('');
  logArrow('🧬 Installing optional VGGT ("twin") dependencies');
  log('');
  execFileSync('node', [twinScript, '--twin'], { cwd: rootDir, stdio: 'inherit' });
}

async function main() {
  console.log(banner);
  
  try {
    await installDependencies();
    await setupPythonBackendEnvironment();
    const backendDepsInstalled = await installBackendDeps();
    if (!backendDepsInstalled) {
      throw new Error('Backend dependencies installation failed');
    }
    const lerobotToolchainInstalled = await installOfficialLeRobotToolchain();
    if (!lerobotToolchainInstalled) {
      throw new Error('Official LeRobot dataset toolchain installation failed');
    }
    const openArmHardwareRuntimeInstalled = await installOpenArmHardwareRuntime();
    if (!openArmHardwareRuntimeInstalled) {
      throw new Error('OpenArm hardware runtime installation failed');
    }
    const mjlabRuntimeInstalled = await installMjlabRuntime();
    if (!mjlabRuntimeInstalled) {
      throw new Error('MJLab validation runtime installation failed');
    }
    await installTwinDepsIfRequested();
    await checkIkd();
    await setupHuggingFace();
    await setupGitHub();
    const globalIluResult = await installOptionalGlobalIlu();
    
    log('');
    logSuccess('Setup complete');
    printSetupSummary({ globalIluResult });
  } catch (error) {
    log('');
    log('✗ Setup failed', colors.yellow);
    process.exit(1);
  }
}

main();
