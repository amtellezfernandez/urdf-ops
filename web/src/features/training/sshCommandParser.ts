export interface ParsedSshCommand {
  user?: string;
  host: string;
  port?: number;
}

const SSH_OPTIONS_WITH_VALUE = new Set(["-b", "-c", "-D", "-E", "-F", "-i", "-J", "-L", "-l", "-m", "-O", "-o", "-p", "-Q", "-R", "-S", "-W"]);

function parsePort(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 65535 ? parsed : undefined;
}

function normalizeTarget(value: string): string {
  return value
    .trim()
    .replace(/^ssh:\/\//i, "")
    .replace(/^\/+|\/+$/g, "");
}

export function parseSshCommand(value: string): ParsedSshCommand | null {
  const tokens = value.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return null;

  let index = tokens[0] === "ssh" ? 1 : 0;
  let port: number | undefined;
  let userFromOption: string | undefined;
  let target: string | undefined;

  while (index < tokens.length) {
    const token = tokens[index];

    if (token === "-p") {
      port = parsePort(tokens[index + 1]);
      index += 2;
      continue;
    }

    if (token.startsWith("-p") && token.length > 2) {
      port = parsePort(token.slice(2));
      index += 1;
      continue;
    }

    if (token === "-l") {
      userFromOption = tokens[index + 1];
      index += 2;
      continue;
    }

    if (token.startsWith("-")) {
      index += SSH_OPTIONS_WITH_VALUE.has(token) ? 2 : 1;
      continue;
    }

    target = token;
    break;
  }

  if (!target) return null;
  const normalizedTarget = normalizeTarget(target);
  const targetMatch = normalizedTarget.match(/^(?:(?<user>[^@\s]+)@)?(?<host>[^@\s:]+)(?::(?<port>\d+))?$/);
  const host = targetMatch?.groups?.host;
  if (!host) return null;

  return {
    user: targetMatch.groups?.user || userFromOption,
    host,
    port: parsePort(targetMatch.groups?.port) || port,
  };
}
