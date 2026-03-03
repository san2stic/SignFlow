import { useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Wifi, WifiOff, AlertCircle } from "lucide-react";
import { getStatus, getHistory } from "../api/updater";
import { useUpdaterStore } from "../stores/updaterStore";
import { StatusBadge } from "../components/updater/StatusBadge";
import { GitStatusPanel } from "../components/updater/GitStatusPanel";
import { BuildLogViewer } from "../components/updater/BuildLogViewer";
import { DeploymentHistoryTable } from "../components/updater/DeploymentHistoryTable";

// Animations cohérentes avec le reste de l'app
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

// Backoff exponentiel : 1s → 2s → 4s → 8s → 16s → 60s max
function computeBackoff(attempt: number): number {
  return Math.min(1000 * Math.pow(2, attempt), 60_000);
}

function buildWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${proto}//${host}/api/v1/updater/live`;
}

interface WsHeartbeat {
  type: "heartbeat";
  ts: string;
  state: string;
  local_commit?: string;
  remote_commit?: string;
}
interface WsStatusUpdate {
  type: "status_update";
  ts: string;
  state: string;
  deployment_id: number;
  commit_hash?: string;
}
interface WsBuildLog {
  type: "build_log";
  ts: string;
  deployment_id: number;
  line: string;
}
interface WsCompleted {
  type: "completed";
  ts: string;
  deployment_id: number;
  commit_hash: string;
  duration_s: number;
}
interface WsError {
  type: "error";
  ts: string;
  deployment_id: number;
  message: string;
}
interface WsPong {
  type: "pong";
  ts: string;
}

type WsMessage = WsHeartbeat | WsStatusUpdate | WsBuildLog | WsCompleted | WsError | WsPong;

export function UpdaterPage(): JSX.Element {
  const {
    status,
    history,
    buildLogLines,
    wsConnected,
    isLoading,
    error,
    setStatus,
    setHistory,
    appendLogLine,
    clearLogs,
    setWsConnected,
    setLoading,
    setError
  } = useUpdaterStore();

  const wsRef = useRef<WebSocket | null>(null);
  const retryAttemptRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  // Chargement initial des données REST
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [statusData, historyData] = await Promise.allSettled([
        getStatus(),
        getHistory(20)
      ]);
      if (statusData.status === "fulfilled") setStatus(statusData.value);
      if (historyData.status === "fulfilled") setHistory(historyData.value);
    } catch {
      // silencieux, erreurs gérées par allSettled
    } finally {
      setLoading(false);
    }
  }, [setStatus, setHistory, setLoading]);

  // Connexion WebSocket avec reconnexion automatique
  const connectWs = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(buildWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setWsConnected(true);
      retryAttemptRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      if (!mountedRef.current) return;
      let msg: WsMessage;
      try {
        msg = JSON.parse(event.data) as WsMessage;
      } catch {
        return;
      }

      switch (msg.type) {
        case "heartbeat":
          setWsConnected(true);
          // Recharger le statut complet depuis REST au heartbeat uniquement si inactif
          // pour éviter les re-renders en boucle pendant un build
          break;

        case "status_update": {
          // Effacer les logs si on démarre un nouveau pipeline depuis un état terminal
          const prevState = status?.state;
          if (
            prevState === "idle" ||
            prevState === "success" ||
            prevState === "error" ||
            prevState === "rolled_back"
          ) {
            clearLogs();
          }
          // Rafraîchir le statut complet depuis l'API REST
          void loadData();
          break;
        }

        case "build_log":
          appendLogLine(msg.line);
          break;

        case "completed":
          // Recharger statut + historique après complétion
          void loadData();
          break;

        case "error":
          setError(msg.message);
          // Recharger le statut depuis l'API REST
          void loadData();
          break;

        default:
          break;
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setWsConnected(false);
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setWsConnected(false);
      // Reconnexion avec backoff exponentiel
      const delay = computeBackoff(retryAttemptRef.current);
      retryAttemptRef.current += 1;
      retryTimerRef.current = setTimeout(() => {
        if (mountedRef.current) {
          connectWs();
        }
      }, delay);
    };
  }, [status?.state, setStatus, setWsConnected, setError, clearLogs, appendLogLine, loadData]);

  useEffect(() => {
    mountedRef.current = true;
    void loadData();
    connectWs();

    return () => {
      mountedRef.current = false;
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRefreshAfterTrigger = useCallback(() => {
    void loadData();
  }, [loadData]);

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="relative space-y-6 pb-8"
    >
      {/* Header */}
      <motion.header variants={itemVariants} className="flex flex-wrap items-center gap-4">
        <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
          <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
            <RefreshCw className="h-6 w-6 text-primary" />
          </div>
        </div>
        <div className="flex-1">
          <h1 className="font-display text-3xl font-bold tracking-tight">
            <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Déploiement Automatique
            </span>
          </h1>
          <p className="text-sm text-text-secondary">Monitoring Git → Docker CI/CD</p>
        </div>

        {/* Statut global + WS indicator */}
        <div className="flex items-center gap-3 flex-wrap">
          {status && <StatusBadge status={status.state} size="lg" />}
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ring-1 transition-all duration-200 ${
              wsConnected
                ? "bg-emerald-900/30 text-emerald-300 ring-emerald-500/30"
                : "bg-slate-700/40 text-slate-400 ring-slate-600/40"
            }`}
          >
            {wsConnected ? (
              <Wifi className="h-3.5 w-3.5 animate-pulse" />
            ) : (
              <WifiOff className="h-3.5 w-3.5" />
            )}
            {wsConnected ? "Connecté" : "Déconnecté"}
          </span>
        </div>
      </motion.header>

      {/* Error banner */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="card border-red-500/30 p-4"
        >
          <div className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-400" />
            <p className="text-sm text-red-300">{error}</p>
            <button
              className="ml-auto text-xs text-slate-400 hover:text-slate-200 transition-colors"
              onClick={() => setError(null)}
            >
              ✕ Fermer
            </button>
          </div>
        </motion.div>
      )}

      {/* Loading spinner (chargement initial) */}
      {isLoading && !status && (
        <motion.div variants={itemVariants} className="card flex min-h-[200px] items-center justify-center">
          <div className="text-center space-y-3">
            <div className="mx-auto h-10 w-10 animate-spin rounded-full border-b-2 border-primary" />
            <p className="text-sm text-text-secondary">Chargement du statut…</p>
          </div>
        </motion.div>
      )}

      {/* Main content grid */}
      {(!isLoading || status) && (
        <>
          {/* Row 1: GitStatusPanel + BuildLogViewer */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-1 gap-5 lg:grid-cols-2"
          >
            <GitStatusPanel status={status} onTrigger={handleRefreshAfterTrigger} />
            <BuildLogViewer lines={buildLogLines} />
          </motion.div>

          {/* Row 2: DeploymentHistoryTable (full width) */}
          <motion.div variants={itemVariants}>
            <DeploymentHistoryTable history={history} onRollback={handleRefreshAfterTrigger} />
          </motion.div>
        </>
      )}
    </motion.section>
  );
}
