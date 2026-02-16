#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"
TAILSCALE_APP_BIN="/Applications/Tailscale.app/Contents/MacOS/Tailscale"

print_usage() {
  cat <<'EOF'
Usage: scripts/expose-signflow-tailscale.sh [start|status|stop]

Commands:
  start   Start SignFlow production services and expose app/API + MLflow via Tailscale (default)
  status  Show Docker and Tailscale exposure status
  stop    Disable Tailscale HTTPS exposure (tailscale serve reset for all exposed ports)
EOF
}

find_tailscale_bin() {
  if command -v tailscale >/dev/null 2>&1; then
    command -v tailscale
    return 0
  fi

  if [ -x "$TAILSCALE_APP_BIN" ]; then
    echo "$TAILSCALE_APP_BIN"
    return 0
  fi

  return 1
}

require_tailscale_running() {
  local state
  state="$("$TS_BIN" status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("BackendState",""))' || true)"
  if [ "$state" != "Running" ]; then
    echo "Tailscale is not running. Open Tailscale and connect this machine first." >&2
    exit 1
  fi
}

wait_for_local_gateway() {
  local tries=60
  until curl -fsS http://127.0.0.1/healthz >/dev/null 2>&1; do
    tries=$((tries - 1))
    if [ "$tries" -le 0 ]; then
      echo "SignFlow is not reachable on http://127.0.0.1/healthz after waiting." >&2
      echo "Check: docker compose -f docker-compose.prod.yml logs caddy backend" >&2
      exit 1
    fi
    sleep 2
  done
}

wait_for_mlflow() {
  local tries=60
  until curl -fsS http://127.0.0.1:5001 >/dev/null 2>&1; do
    tries=$((tries - 1))
    if [ "$tries" -le 0 ]; then
      echo "MLflow is not reachable on http://127.0.0.1:5001 after waiting." >&2
      echo "Check: docker compose -f docker-compose.prod.yml logs mlflow" >&2
      exit 1
    fi
    sleep 2
  done
}

serve_summary() {
  local dns_name serve_routes
  dns_name="$("$TS_BIN" status --json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("Self",{}).get("DNSName","").rstrip("."))' || true)"
  serve_routes="$("$TS_BIN" serve status --json 2>/dev/null | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for host, cfg in sorted(d.get("Web", {}).items()):
    name, _, port = host.partition(":")
    base = f"https://{name}"
    if port and port != "443":
        base = f"{base}:{port}"
    for path, handler in sorted(cfg.get("Handlers", {}).items()):
        proxy = handler.get("Proxy", "")
        if proxy:
            p = path if path.startswith("/") else f"/{path}"
            print(f"{base}{p} -> {proxy}")' || true)"

  echo
  if [ -n "$dns_name" ]; then
    echo "Tailnet app URL: https://$dns_name/"
    echo "Tailnet MLflow URL: https://$dns_name:5001/"
  fi
  if [ -n "$serve_routes" ]; then
    echo "Serve routes:"
    while IFS= read -r route; do
      [ -n "$route" ] && echo "  - $route"
    done <<<"$serve_routes"
  else
    echo "Serve routes: not configured"
  fi
}

start() {
  require_tailscale_running

  echo "Starting SignFlow services (caddy, frontend, backend, celery_worker, mlflow)..."
  docker compose -f "$COMPOSE_FILE" up -d caddy frontend backend celery_worker mlflow

  echo "Waiting for local gateway..."
  wait_for_local_gateway

  echo "Waiting for MLflow..."
  wait_for_mlflow

  echo "Configuring Tailscale HTTPS proxies on ports 443 (app/API) and 5001 (MLflow)..."
  "$TS_BIN" serve --https=443 off >/dev/null 2>&1 || true
  "$TS_BIN" serve --https=5001 off >/dev/null 2>&1 || true
  "$TS_BIN" serve --bg --https=443 --yes http://127.0.0.1:80 >/dev/null
  "$TS_BIN" serve --bg --https=5001 --yes http://127.0.0.1:5001 >/dev/null

  echo "Done."
  serve_summary
}

status() {
  echo "Docker services:"
  docker compose -f "$COMPOSE_FILE" ps caddy frontend backend celery_worker mlflow

  echo
  echo "Tailscale serve config:"
  "$TS_BIN" serve status --json 2>/dev/null || echo "{}"
  serve_summary
}

stop() {
  require_tailscale_running
  echo "Disabling Tailscale HTTPS exposure..."
  "$TS_BIN" serve reset >/dev/null
  echo "Done."
}

TS_BIN="$(find_tailscale_bin || true)"
if [ -z "${TS_BIN:-}" ]; then
  echo "Tailscale CLI not found. Install Tailscale or use the macOS app." >&2
  exit 1
fi

case "${1:-start}" in
  start)
    start
    ;;
  status)
    status
    ;;
  stop)
    stop
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    print_usage >&2
    exit 1
    ;;
esac
