"use client";

import { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:8080";
const WS_URL = "ws://localhost:8080/ws";

interface PhaseData {
  name: string;
  status: "pending" | "running" | "completed" | "failed";
  startTime?: string;
  endTime?: string;
  duration?: string;
  output?: string;
  details?: Record<string, unknown>;
}

interface IterationData {
  index: number;
  status: "running" | "success" | "failed";
  experimentSpec?: Record<string, unknown>;
  evaluationResult?: Record<string, unknown>;
  classification?: string;
  pValue?: number;
  effectSize?: number;
  feedback?: string;
}

export default function Home() {
  const [goal, setGoal] = useState("");
  const [loading, setLoading] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "failed">("idle");
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [expandedPhase, setExpandedPhase] = useState<string | null>(null);
  const [iterations, setIterations] = useState<IterationData[]>([]);
  const [selectedIteration, setSelectedIteration] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [phases, setPhases] = useState<PhaseData[]>([
    { name: "Research", status: "pending" },
    { name: "Design", status: "pending" },
    { name: "Execute", status: "pending" },
    { name: "Evaluate", status: "pending" },
  ]);

  const addLog = (msg: string, type: "info" | "success" | "error" | "warning" = "info") => {
    const icon = type === "success" ? "✓" : type === "error" ? "✗" : type === "warning" ? "⚠" : "›";
    setLogs(prev => [...prev.slice(-100), `${new Date().toLocaleTimeString()} ${icon} ${msg}`]);
  };

  const updatePhase = (name: string, updates: Partial<PhaseData>) => {
    setPhases(prev => prev.map(p => p.name === name ? { ...p, ...updates } : p));
  };

  const startResearch = async () => {
    if (!goal.trim()) return;
    setLoading(true);
    setLogs([]);
    setStatus("running");
    setCurrentAgent(null);
    setIterations([]);
    setPhases(phases.map(p => ({ ...p, status: "pending" as const })));

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      addLog("Connected to Epsilon server", "success");
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);

        switch (data.type) {
          case "connected":
            addLog("WebSocket ready", "success");
            break;
          case "run_started":
            addLog(`Research initiated: "${data.goal}"`, "info");
            break;
          case "agent_started":
            setCurrentAgent(data.agent);
            updatePhase(data.agent, { status: "running", startTime: new Date().toISOString() });
            addLog(`${data.agent} Agent activated`, "info");
            break;
          case "agent_completed":
            updatePhase(data.agent, {
              status: "completed",
              endTime: new Date().toISOString(),
              output: data.output || null,
              details: data.details || null
            });
            addLog(`${data.agent} Agent completed`, "success");
            break;
          case "iteration_started":
            const iterNum = (data.iteration ?? 0) + 1;
            setIterations(prev => [...prev, {
              index: data.iteration ?? prev.length,
              status: "running"
            }]);
            addLog(`Iteration ${iterNum} started`, "info");
            break;
          case "iteration_completed":
            setIterations(prev => prev.map((iter, i) =>
              i === prev.length - 1 ? {
                ...iter,
                status: "success" as const,
                classification: data.classification,
                experimentSpec: data.experiment_spec,
                evaluationResult: data.evaluation,
                pValue: data.p_value,
                effectSize: data.effect_size
              } : iter
            ));
            addLog(`Iteration completed: ${data.classification || "done"}`, "success");
            break;
          case "iteration_failed":
            setIterations(prev => prev.map((iter, i) =>
              i === prev.length - 1 ? { ...iter, status: "failed" as const, feedback: data.message } : iter
            ));
            addLog(`Iteration failed: ${data.message}`, "warning");
            break;
          case "run_completed":
            setStatus("completed");
            setCurrentAgent(null);
            addLog("Research completed successfully!", "success");
            break;
          case "run_error":
          case "error":
            setStatus("failed");
            addLog(`Error: ${data.error}`, "error");
            break;
        }
      } catch { }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    try {
      const res = await fetch(`${API_URL}/api/research/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: goal.trim(), max_iterations: 5 }),
      });

      if (res.ok) {
        const data = await res.json();
        setRunId(data.run_id);
        addLog(`Run ID: ${data.run_id.substring(0, 8)}...`, "info");
      } else {
        addLog("Failed to start research", "error");
        setStatus("failed");
      }
    } catch (err) {
      addLog(`Connection error: ${err}`, "error");
      setStatus("failed");
    }

    setLoading(false);
  };

  const phaseIcons: Record<string, string> = {
    Research: "🔬",
    Design: "📐",
    Execute: "⚡",
    Evaluate: "📊"
  };

  const phaseDescriptions: Record<string, string> = {
    Research: "Gathering relevant papers, evidence, and prior knowledge",
    Design: "Creating statistically rigorous experiment specification",
    Execute: "Running the experiment code with real data",
    Evaluate: "Analyzing results with hypothesis testing"
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xl">
              ε
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Epsilon</h1>
              <p className="text-xs text-slate-400">Autonomous Research Engine</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {runId && (
              <span className="text-xs font-mono text-slate-500">
                Run: {runId.substring(0, 8)}
              </span>
            )}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${connected
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
              }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`} />
              {connected ? "Connected" : "Disconnected"}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Research Input */}
        <section className="mb-8">
          <div className="bg-slate-900/50 backdrop-blur border border-slate-800/50 rounded-2xl p-6">
            <label className="block text-sm font-medium text-slate-400 mb-3">
              Research Question
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="e.g., Does dropout regularization improve Transformer training stability?"
                className="flex-1 bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white placeholder:text-slate-500 focus:outline-none focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                disabled={status === "running"}
                onKeyDown={(e) => e.key === "Enter" && startResearch()}
              />
              <button
                onClick={startResearch}
                disabled={loading || status === "running" || !goal.trim()}
                className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-medium text-white transition-all shadow-lg shadow-indigo-500/25"
              >
                {status === "running" ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Running...
                  </span>
                ) : "Start Research"}
              </button>
            </div>

            {/* Quick Examples */}
            {status === "idle" && (
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="text-xs text-slate-500">Examples:</span>
                {[
                  "Does dropout improve Transformer stability?",
                  "Effect of learning rate warmup on training",
                ].map((ex) => (
                  <button
                    key={ex}
                    onClick={() => setGoal(ex)}
                    className="text-xs px-3 py-1 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 rounded-lg text-slate-400 hover:text-white transition-all"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Main Content Grid */}
        {status !== "idle" && (
          <div className="grid grid-cols-12 gap-6">
            {/* Left Column - Pipeline & Phases */}
            <div className="col-span-8 space-y-6">
              {/* Status Banner */}
              <div className={`rounded-xl p-4 border ${status === "completed"
                  ? "bg-emerald-500/10 border-emerald-500/20"
                  : status === "failed"
                    ? "bg-rose-500/10 border-rose-500/20"
                    : "bg-indigo-500/10 border-indigo-500/20"
                }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {status === "running" && (
                      <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                        <svg className="animate-spin h-4 w-4 text-indigo-400" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                      </div>
                    )}
                    <div>
                      <h2 className={`font-semibold ${status === "completed" ? "text-emerald-400" :
                          status === "failed" ? "text-rose-400" : "text-indigo-400"
                        }`}>
                        {status === "completed" && "✓ Research Complete"}
                        {status === "failed" && "✗ Research Failed"}
                        {status === "running" && "Research in Progress"}
                      </h2>
                      <p className="text-sm text-slate-400 mt-0.5">
                        {goal}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-white">
                      {iterations.length}
                    </div>
                    <div className="text-xs text-slate-400">Iterations</div>
                  </div>
                </div>
              </div>

              {/* Phase Cards */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                  Research Pipeline
                </h3>

                {phases.map((phase, idx) => (
                  <div
                    key={phase.name}
                    className={`bg-slate-900/50 backdrop-blur border rounded-xl overflow-hidden transition-all ${phase.status === "running"
                        ? "border-indigo-500/50 shadow-lg shadow-indigo-500/10"
                        : phase.status === "completed"
                          ? "border-emerald-500/30"
                          : "border-slate-800/50"
                      }`}
                  >
                    {/* Phase Header */}
                    <div
                      className="p-4 cursor-pointer hover:bg-slate-800/30 transition-all"
                      onClick={() => setExpandedPhase(expandedPhase === phase.name ? null : phase.name)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-2xl ${phase.status === "running"
                              ? "bg-indigo-500/20 animate-pulse"
                              : phase.status === "completed"
                                ? "bg-emerald-500/20"
                                : "bg-slate-800/50"
                            }`}>
                            {phaseIcons[phase.name]}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-semibold text-white">{phase.name} Agent</h4>
                              {phase.status === "running" && (
                                <span className="text-xs px-2 py-0.5 bg-indigo-500/20 text-indigo-400 rounded-full">
                                  Active
                                </span>
                              )}
                              {phase.status === "completed" && (
                                <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full">
                                  ✓ Done
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-slate-400 mt-0.5">
                              {phaseDescriptions[phase.name]}
                            </p>
                          </div>
                        </div>
                        <svg
                          className={`w-5 h-5 text-slate-400 transition-transform ${expandedPhase === phase.name ? "rotate-180" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    </div>

                    {/* Phase Details */}
                    {expandedPhase === phase.name && (
                      <div className="border-t border-slate-800/50 p-4 bg-slate-950/30">
                        {phase.status === "pending" ? (
                          <p className="text-sm text-slate-500 italic">Waiting to start...</p>
                        ) : (
                          <div className="space-y-4">
                            {/* Status Info */}
                            <div className="grid grid-cols-3 gap-4">
                              <div className="bg-slate-800/30 rounded-lg p-3">
                                <div className="text-xs text-slate-500 mb-1">Status</div>
                                <div className={`text-sm font-medium ${phase.status === "completed" ? "text-emerald-400" :
                                    phase.status === "running" ? "text-indigo-400" : "text-slate-400"
                                  }`}>
                                  {phase.status.charAt(0).toUpperCase() + phase.status.slice(1)}
                                </div>
                              </div>
                              {phase.startTime && (
                                <div className="bg-slate-800/30 rounded-lg p-3">
                                  <div className="text-xs text-slate-500 mb-1">Started</div>
                                  <div className="text-sm font-medium text-white">
                                    {new Date(phase.startTime).toLocaleTimeString()}
                                  </div>
                                </div>
                              )}
                              {phase.endTime && (
                                <div className="bg-slate-800/30 rounded-lg p-3">
                                  <div className="text-xs text-slate-500 mb-1">Duration</div>
                                  <div className="text-sm font-medium text-white">
                                    {Math.round((new Date(phase.endTime).getTime() - new Date(phase.startTime!).getTime()) / 1000)}s
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Output Preview */}
                            {phase.output && (
                              <div>
                                <div className="text-xs text-slate-500 mb-2">Output</div>
                                <div className="bg-slate-950/50 rounded-lg p-3 font-mono text-xs text-slate-300 max-h-40 overflow-y-auto">
                                  {phase.output}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Iterations Report */}
              {iterations.length > 0 && (
                <div className="bg-slate-900/50 backdrop-blur border border-slate-800/50 rounded-xl p-6">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
                    Experiment Iterations
                  </h3>

                  <div className="space-y-3">
                    {iterations.map((iter, idx) => (
                      <div
                        key={idx}
                        className={`border rounded-lg p-4 cursor-pointer transition-all ${selectedIteration === idx
                            ? "border-indigo-500/50 bg-indigo-500/5"
                            : iter.status === "success"
                              ? "border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10"
                              : iter.status === "failed"
                                ? "border-rose-500/30 bg-rose-500/5 hover:bg-rose-500/10"
                                : "border-slate-700/50 hover:bg-slate-800/30"
                          }`}
                        onClick={() => setSelectedIteration(selectedIteration === idx ? null : idx)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold ${iter.status === "success" ? "bg-emerald-500/20 text-emerald-400" :
                                iter.status === "failed" ? "bg-rose-500/20 text-rose-400" :
                                  "bg-indigo-500/20 text-indigo-400"
                              }`}>
                              {idx + 1}
                            </div>
                            <div>
                              <div className="font-medium text-white">Iteration {idx + 1}</div>
                              {iter.classification && (
                                <div className="text-xs text-slate-400">
                                  Classification: <span className={`font-medium ${iter.classification === "robust" ? "text-emerald-400" : "text-amber-400"
                                    }`}>{iter.classification}</span>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Quick Stats */}
                          {iter.pValue !== undefined && (
                            <div className="flex gap-4 text-right">
                              <div>
                                <div className="text-xs text-slate-500">p-value</div>
                                <div className={`text-sm font-mono font-medium ${iter.pValue < 0.05 ? "text-emerald-400" : "text-amber-400"
                                  }`}>
                                  {iter.pValue.toFixed(4)}
                                </div>
                              </div>
                              {iter.effectSize !== undefined && (
                                <div>
                                  <div className="text-xs text-slate-500">Effect Size</div>
                                  <div className="text-sm font-mono font-medium text-white">
                                    {iter.effectSize.toFixed(3)}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Expanded Details */}
                        {selectedIteration === idx && (
                          <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-4">
                            {iter.experimentSpec && (
                              <div>
                                <div className="text-xs text-slate-500 mb-2">Experiment Specification</div>
                                <pre className="bg-slate-950/50 rounded-lg p-3 font-mono text-xs text-slate-300 overflow-x-auto">
                                  {JSON.stringify(iter.experimentSpec, null, 2)}
                                </pre>
                              </div>
                            )}
                            {iter.evaluationResult && (
                              <div>
                                <div className="text-xs text-slate-500 mb-2">Evaluation Results</div>
                                <pre className="bg-slate-950/50 rounded-lg p-3 font-mono text-xs text-slate-300 overflow-x-auto">
                                  {JSON.stringify(iter.evaluationResult, null, 2)}
                                </pre>
                              </div>
                            )}
                            {iter.feedback && (
                              <div>
                                <div className="text-xs text-slate-500 mb-2">Feedback</div>
                                <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-sm text-amber-200">
                                  {iter.feedback}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Logs */}
            <div className="col-span-4 space-y-6">
              {/* Live Logs */}
              <div className="bg-slate-900/50 backdrop-blur border border-slate-800/50 rounded-xl p-4 sticky top-24">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                    Live Activity Log
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-xs text-slate-500">Live</span>
                  </div>
                </div>

                <div className="h-[600px] overflow-y-auto space-y-1 font-mono text-xs">
                  {logs.length === 0 ? (
                    <p className="text-slate-500 italic">Waiting for activity...</p>
                  ) : (
                    logs.map((log, i) => (
                      <div
                        key={i}
                        className={`py-1 px-2 rounded transition-colors ${log.includes("✓") ? "text-emerald-400 bg-emerald-500/5" :
                            log.includes("✗") ? "text-rose-400 bg-rose-500/5" :
                              log.includes("⚠") ? "text-amber-400 bg-amber-500/5" :
                                "text-slate-400 hover:bg-slate-800/30"
                          }`}
                      >
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
