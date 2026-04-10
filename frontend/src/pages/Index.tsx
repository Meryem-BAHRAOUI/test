import React, { useState, useRef, useCallback, useEffect } from "react";
import { Monitor, RefreshCw, Download } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import ChatBubble from "@/components/ChatBubble";
import ChatInput from "@/components/ChatInput";

const BACKEND_URL = "http://127.0.0.1:8002";
const COUNTDOWN_SEC = 5;

// ── Types ─────────────────────────────────────────────────────────────────────

interface UiElement {
  id: string;
  bbox: [number, number, number, number];
  score: number;
}

interface OcrBlock {
  id: string;
  text: string;
  bbox: [number, number, number, number];
  conf: number;
}

interface AnalysisResult {
  screen_size: [number, number];
  timestamp: string;
  ui_elements: UiElement[];
  ocr_blocks: OcrBlock[];
  image_b64: string;
  original_b64: string;
  json_file: string;
}

interface ResolvedAction {
  action: string;
  target_text: string;
  value?: string;
  matched_text_bbox: [number, number, number, number];
  resolved_element_id: string | null;
  resolved_bbox: [number, number, number, number];
  resolved_point: [number, number];
  status: "resolved";
}

interface UnresolvedAction {
  action: string;
  target_text: string;
  value?: string;
  status: "unresolved";
  reason: string;
}

interface ResolutionResult {
  resolved_actions: ResolvedAction[];
  unresolved_actions: UnresolvedAction[];
  generated_actions: { action: string; target_text: string; value?: string }[];
  llm_metadata: Record<string, unknown>;
}

// ── Chat message union ─────────────────────────────────────────────────────────

type ChatMsg =
  | { id: string; role: "user"; text: string; ts: string }
  | { id: string; role: "ai"; kind: "text";       text: string;             ts: string }
  | { id: string; role: "ai"; kind: "capture";    result: AnalysisResult;   ts: string }
  | { id: string; role: "ai"; kind: "resolution"; data: ResolutionResult; original_b64: string; ts: string }
  | { id: string; role: "ai"; kind: "loading";    label: string;            ts: string }
  | { id: string; role: "ai"; kind: "error";      text: string;             ts: string };

// ── Helpers ───────────────────────────────────────────────────────────────────

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function uid() {
  return Math.random().toString(36).slice(2);
}

const ACTION_COLORS: Record<string, string> = {
  fill:    "bg-blue-100 text-blue-700 border-blue-300",
  click:   "bg-purple-100 text-purple-700 border-purple-300",
  select:  "bg-orange-100 text-orange-700 border-orange-300",
  check:   "bg-teal-100 text-teal-700 border-teal-300",
  uncheck: "bg-gray-100 text-gray-700 border-gray-300",
};

// Couleurs SVG par type d'action pour l'overlay de résolution
const ACTION_HEX: Record<string, string> = {
  fill:    "#3b82f6",  // bleu
  click:   "#a855f7",  // violet
  select:  "#f97316",  // orange
  check:   "#14b8a6",  // teal
  uncheck: "#6b7280",  // gris
};

function ActionTag({ action }: { action: string }) {
  const cls = ACTION_COLORS[action] ?? "bg-gray-100 text-gray-700 border-gray-300";
  return (
    <span className={`inline-block text-[11px] font-mono font-bold px-1.5 py-0.5 rounded border ${cls}`}>
      {action}
    </span>
  );
}

// ── SVG overlay ───────────────────────────────────────────────────────────────

function BboxOverlay({ elements, ocr, imgW, imgH, displayW, displayH }: {
  elements: UiElement[]; ocr: OcrBlock[];
  imgW: number; imgH: number; displayW: number; displayH: number;
}) {
  if (!imgW || !imgH || !displayW || !displayH) return null;
  const sx = displayW / imgW;
  const sy = displayH / imgH;
  return (
    <svg className="absolute inset-0 pointer-events-none" width={displayW} height={displayH}>
      {elements.map((el) => {
        const [x1, y1, x2, y2] = el.bbox;
        return (
          <g key={el.id}>
            <rect x={x1*sx} y={y1*sy} width={(x2-x1)*sx} height={(y2-y1)*sy}
              fill="none" stroke="#3b82f6" strokeWidth="2" />
            <text x={x1*sx+2} y={Math.max(y1*sy-3, 12)} fill="#3b82f6" fontSize="11" fontWeight="bold">
              {el.id} {el.score.toFixed(2)}
            </text>
          </g>
        );
      })}
      {ocr.map((bl) => {
        const [x1, y1, x2, y2] = bl.bbox;
        return (
          <g key={bl.id}>
            <rect x={x1*sx} y={y1*sy} width={(x2-x1)*sx} height={(y2-y1)*sy}
              fill="rgba(0,200,80,0.08)" stroke="#00c850" strokeWidth="1.5" />
            <text x={x1*sx+2} y={Math.max(y1*sy-3, 12)} fill="#00c850" fontSize="10">
              {bl.text.length > 20 ? bl.text.slice(0,20)+"…" : bl.text}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Resolution overlay SVG ────────────────────────────────────────────────────

function ResolutionOverlay({ actions, imgW, imgH, displayW, displayH }: {
  actions: ResolvedAction[];
  imgW: number; imgH: number; displayW: number; displayH: number;
}) {
  if (!imgW || !imgH || !displayW || !displayH) return null;
  const sx = displayW / imgW;
  const sy = displayH / imgH;

  return (
    <svg className="absolute inset-0 pointer-events-none" width={displayW} height={displayH}>
      {actions.map((a, i) => {
        const color = ACTION_HEX[a.action] ?? "#ef4444";
        const [tx1, ty1, tx2, ty2] = a.matched_text_bbox;
        const [rx1, ry1, rx2, ry2] = a.resolved_bbox;
        const [px, py] = a.resolved_point;
        const label = a.value ? `${i + 1}. ${a.action}: ${a.value}` : `${i + 1}. ${a.action}`;

        return (
          <g key={i}>
            {/* Texte matché — tirets fins */}
            <rect
              x={tx1 * sx} y={ty1 * sy}
              width={(tx2 - tx1) * sx} height={(ty2 - ty1) * sy}
              fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="4 3" opacity="0.8"
            />

            {/* Élément résolu — plein + fond semi-transparent */}
            <rect
              x={rx1 * sx} y={ry1 * sy}
              width={(rx2 - rx1) * sx} height={(ry2 - ry1) * sy}
              fill={color} fillOpacity="0.15" stroke={color} strokeWidth="2.5"
            />

            {/* Numéro dans coin haut-gauche */}
            <rect
              x={rx1 * sx} y={ry1 * sy - 16}
              width={label.length * 6.5 + 8} height={16}
              fill={color} rx="3"
            />
            <text
              x={rx1 * sx + 4} y={ry1 * sy - 4}
              fill="white" fontSize="10" fontWeight="bold"
            >
              {label}
            </text>

            {/* Point de clic — croix */}
            <line x1={(px - 6) * sx} y1={py * sy} x2={(px + 6) * sx} y2={py * sy}
              stroke={color} strokeWidth="2" />
            <line x1={px * sx} y1={(py - 6) * sy} x2={px * sx} y2={(py + 6) * sy}
              stroke={color} strokeWidth="2" />
            <circle cx={px * sx} cy={py * sy} r="4"
              fill="none" stroke={color} strokeWidth="2" />
          </g>
        );
      })}
    </svg>
  );
}

// ── Countdown overlay ─────────────────────────────────────────────────────────

function CountdownOverlay({ seconds }: { seconds: number }) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/75">
      <div className="text-[10rem] font-black text-white leading-none drop-shadow-xl">{seconds}</div>
      <p className="text-white text-2xl mt-6 font-semibold">Switch to the window you want to capture</p>
      <p className="text-gray-400 text-base mt-2">Screenshot will be taken automatically</p>
    </div>
  );
}

// ── Rich AI message renderer ──────────────────────────────────────────────────

function CaptureMessage({ result, ts }: { result: AnalysisResult; ts: string }) {
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [imgSize, setImgSize]   = useState({ w: 0, h: 0 });
  const [dispSize, setDispSize] = useState({ w: 0, h: 0 });

  const onLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight });
    setDispSize({ w: e.currentTarget.clientWidth,  h: e.currentTarget.clientHeight  });
  }, []);

  const src = `data:image/png;base64,${showAnnotated ? result.image_b64 : result.original_b64}`;

  const downloadJson = () => {
    const clean = {
      screen_size: result.screen_size, timestamp: result.timestamp,
      ui_elements: result.ui_elements, ocr_blocks: result.ocr_blocks,
    };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" }));
    a.download = `analysis_${result.timestamp.replace(/[:.]/g, "-")}.json`;
    a.click();
  };

  return (
    <div className="flex justify-start animate-message-in">
      <div className="max-w-[85%] rounded-2xl px-4 py-3 shadow-sm bg-chat-ai border border-border space-y-3">
        {/* Header */}
        <div className="flex items-center gap-2 flex-wrap">
          <Monitor className="w-4 h-4 text-primary" />
          <span className="font-semibold text-sm">Screen captured</span>
          <Badge variant="outline" className="text-xs">{result.screen_size[0]}×{result.screen_size[1]}</Badge>
          <span className="text-xs text-muted-foreground ml-auto">{ts}</span>
        </div>

        {/* Stats */}
        <div className="flex gap-3">
          <div className="bg-blue-50 rounded-lg px-3 py-1.5 text-center min-w-[70px]">
            <div className="text-xl font-bold text-blue-600">{result.ui_elements.length}</div>
            <div className="text-[10px] text-blue-500">UI elements</div>
          </div>
          <div className="bg-green-50 rounded-lg px-3 py-1.5 text-center min-w-[70px]">
            <div className="text-xl font-bold text-green-600">{result.ocr_blocks.length}</div>
            <div className="text-[10px] text-green-500">Text blocks</div>
          </div>
        </div>

        {/* Thumbnail */}
        <div className="relative inline-block w-full max-w-md">
          <img src={src} alt="capture" className="w-full rounded border" onLoad={onLoad} />
          {!showAnnotated && (
            <BboxOverlay elements={result.ui_elements} ocr={result.ocr_blocks}
              imgW={imgSize.w} imgH={imgSize.h} displayW={dispSize.w} displayH={dispSize.h} />
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => setShowAnnotated(v => !v)}>
            <RefreshCw className="w-3 h-3 mr-1" />
            {showAnnotated ? "Original" : "Annotated"}
          </Button>
          <Button size="sm" variant="outline" onClick={downloadJson}>
            <Download className="w-3 h-3 mr-1" />
            JSON
          </Button>
        </div>

        <p className="text-xs text-muted-foreground italic">
          Now type what you want to do on this screen.
        </p>
      </div>
    </div>
  );
}

function ResolutionMessage({ data, original_b64, ts }: { data: ResolutionResult; original_b64: string; ts: string }) {
  const ok  = data.resolved_actions.length;
  const bad = data.unresolved_actions.length;
  const [showOverlay, setShowOverlay] = useState(true);
  const [imgSize, setImgSize]   = useState({ w: 0, h: 0 });
  const [dispSize, setDispSize] = useState({ w: 0, h: 0 });

  const onLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    setImgSize({ w: e.currentTarget.naturalWidth,  h: e.currentTarget.naturalHeight });
    setDispSize({ w: e.currentTarget.clientWidth,   h: e.currentTarget.clientHeight  });
  }, []);

  return (
    <div className="flex justify-start animate-message-in">
      <div className="max-w-[85%] rounded-2xl px-4 py-3 shadow-sm bg-chat-ai border border-border space-y-3">
        {/* Header */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm">Actions resolved</span>
          {ok  > 0 && <Badge className="text-xs bg-green-600">{ok} resolved</Badge>}
          {bad > 0 && <Badge variant="destructive" className="text-xs">{bad} unresolved</Badge>}
          <span className="text-xs text-muted-foreground ml-auto">{ts}</span>
        </div>

        {/* Image avec overlay de résolution */}
        {ok > 0 && (
          <div className="space-y-2">
            <div className="relative inline-block w-full max-w-lg">
              <img
                src={`data:image/png;base64,${original_b64}`}
                alt="resolution overlay"
                className="w-full rounded border"
                onLoad={onLoad}
              />
              {showOverlay && (
                <ResolutionOverlay
                  actions={data.resolved_actions}
                  imgW={imgSize.w} imgH={imgSize.h}
                  displayW={dispSize.w} displayH={dispSize.h}
                />
              )}
            </div>
            {/* Légende */}
            <div className="flex items-center gap-3 flex-wrap">
              <Button size="sm" variant="outline" onClick={() => setShowOverlay(v => !v)}>
                <RefreshCw className="w-3 h-3 mr-1" />
                {showOverlay ? "Masquer overlay" : "Afficher overlay"}
              </Button>
              <div className="flex gap-2 flex-wrap text-[10px]">
                {data.resolved_actions.map((a, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span className="inline-block w-3 h-2 rounded-sm border-2"
                      style={{ borderColor: ACTION_HEX[a.action] ?? "#ef4444", background: (ACTION_HEX[a.action] ?? "#ef4444") + "26" }} />
                    <span className="font-mono">{i + 1}. {a.target_text}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Generated actions */}
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">
            Parsed from instruction
          </p>
          <div className="flex flex-wrap gap-1.5">
            {data.generated_actions.map((a, i) => (
              <div key={i} className="flex items-center gap-1 bg-muted rounded px-2 py-1 text-xs">
                <ActionTag action={a.action} />
                <span className="font-medium">{a.target_text}</span>
                {a.value && <span className="text-muted-foreground">= {a.value}</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Resolved */}
        {ok > 0 && (
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-green-700 mb-1">Resolved</p>
            {data.resolved_actions.map((a, i) => (
              <div key={i} className="bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-xs space-y-0.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <ActionTag action={a.action} />
                  <span className="font-semibold">{a.target_text}</span>
                  {a.value && <span className="text-muted-foreground">= <code className="bg-white px-1 rounded">{a.value}</code></span>}
                  {a.resolved_element_id && <Badge variant="outline" className="text-[10px]">{a.resolved_element_id}</Badge>}
                </div>
                <div className="text-muted-foreground font-mono">
                  bbox [{a.resolved_bbox.join(", ")}] · click ({a.resolved_point.join(", ")})
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Unresolved */}
        {bad > 0 && (
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-red-600 mb-1">Unresolved</p>
            {data.unresolved_actions.map((a, i) => (
              <div key={i} className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <ActionTag action={a.action} />
                  <span className="font-semibold">{a.target_text}</span>
                  <Badge variant="destructive" className="text-[10px]">{a.reason}</Badge>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Timing */}
        <p className="text-[10px] text-muted-foreground font-mono">
          LLM: {String(data.llm_metadata.wall_time_ms ?? "—")} ms
          {data.llm_metadata.attempt_count ? ` · ${String(data.llm_metadata.attempt_count)} attempt(s)` : ""}
        </p>
      </div>
    </div>
  );
}

function LoadingBubble({ label }: { label: string }) {
  return (
    <div className="flex justify-start animate-message-in">
      <div className="rounded-2xl px-4 py-3 shadow-sm bg-chat-ai border border-border flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
    </div>
  );
}

// ── Page principale ───────────────────────────────────────────────────────────

export default function Index() {
  const { toast } = useToast();
  const [messages, setMessages]         = useState<ChatMsg[]>([
    { id: "welcome", role: "ai", kind: "text",
      text: "Hello! Click the 📷 camera button to capture your screen, then tell me what to do on it.",
      ts: now() },
  ]);
  const [countdown, setCountdown]       = useState<number | null>(null);
  const [busy, setBusy]                 = useState(false);
  const [latestCapture, setLatestCapture] = useState<AnalysisResult | null>(null);

  const fetchStarted = useRef(false);
  const bottomRef    = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Helpers ──────────────────────────────────────────────────────────────────

  const addMsg = (msg: Omit<ChatMsg, "id">) =>
    setMessages(prev => [...prev, { ...msg, id: uid() } as ChatMsg]);

  const replaceLoading = (id: string, replacement: ChatMsg) =>
    setMessages(prev => prev.map(m => m.id === id ? replacement : m));

  // ── Countdown ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (countdown === null) return;
    if (countdown === 0) {
      if (!fetchStarted.current) {
        fetchStarted.current = true;
        doCapture();
      }
      return;
    }
    const t = setTimeout(() => setCountdown(c => (c ?? 1) - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  const handleCapture = () => {
    if (busy || countdown !== null) return;
    fetchStarted.current = false;
    setCountdown(COUNTDOWN_SEC);
  };

  // ── Capture ───────────────────────────────────────────────────────────────────

  const doCapture = async () => {
    setCountdown(null);
    setBusy(true);
    const loadId = uid();
    setMessages(prev => [...prev, {
      id: loadId, role: "ai", kind: "loading", label: "Capturing and analyzing screen…", ts: now(),
    }]);
    try {
      const res = await fetch(`${BACKEND_URL}/analyze/capture`, { method: "POST" });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data: AnalysisResult = await res.json();
      setLatestCapture(data);
      const ts = now();
      replaceLoading(loadId, { id: loadId, role: "ai", kind: "capture", result: data, ts });
    } catch (e: any) {
      replaceLoading(loadId, { id: loadId, role: "ai", kind: "error",
        text: `Capture failed: ${e.message}`, ts: now() });
      toast({ title: "Capture failed", description: e.message, variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  // ── Send instruction ──────────────────────────────────────────────────────────

  const handleSend = async (text: string) => {
    if (busy) return;

    addMsg({ role: "user", text, ts: now() });

    if (!latestCapture) {
      addMsg({ role: "ai", kind: "text",
        text: "No screen captured yet. Please click the 📷 camera button first.", ts: now() });
      return;
    }

    setBusy(true);
    const loadId = uid();
    setMessages(prev => [...prev, {
      id: loadId, role: "ai", kind: "loading", label: "Calling Ollama and resolving actions…", ts: now(),
    }]);

    try {
      const screenData = {
        screen_size:  latestCapture.screen_size,
        timestamp:    latestCapture.timestamp,
        ui_elements:  latestCapture.ui_elements,
        ocr_blocks:   latestCapture.ocr_blocks,
      };
      const res = await fetch(`${BACKEND_URL}/resolve`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ screen_data: screenData, instruction: text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `Server error ${res.status}` }));
        throw new Error(err.detail ?? `Server error ${res.status}`);
      }
      const data: ResolutionResult = await res.json();
      const ts = now();
      replaceLoading(loadId, { id: loadId, role: "ai", kind: "resolution", data, original_b64: latestCapture.original_b64, ts });
    } catch (e: any) {
      replaceLoading(loadId, { id: loadId, role: "ai", kind: "error",
        text: `Resolution failed: ${e.message}`, ts: now() });
      toast({ title: "Resolution failed", description: e.message, variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <>
      {countdown !== null && countdown > 0 && <CountdownOverlay seconds={countdown} />}

      <div className="flex flex-col h-screen bg-background">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-card shrink-0">
          <div>
            <h1 className="text-base font-bold leading-tight">Screen Analyzer</h1>
            <p className="text-xs text-muted-foreground">YOLOX · PaddleOCR · Ollama</p>
          </div>
          <div className="flex gap-1.5 flex-wrap justify-end">
            <Badge variant="outline" className="text-xs">Port 8002</Badge>
            {latestCapture && (
              <Badge className="text-xs bg-green-600">
                {latestCapture.ui_elements.length} UI · {latestCapture.ocr_blocks.length} text
              </Badge>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.map((msg) => {
            if (msg.role === "user") {
              return <ChatBubble key={msg.id} sender="user" content={msg.text} timestamp={msg.ts} />;
            }
            if (msg.kind === "text") {
              return <ChatBubble key={msg.id} sender="ai" content={msg.text} timestamp={msg.ts} />;
            }
            if (msg.kind === "error") {
              return (
                <div key={msg.id} className="flex justify-start animate-message-in">
                  <div className="max-w-[75%] rounded-2xl px-4 py-3 shadow-sm bg-red-50 border border-red-200">
                    <p className="text-sm text-red-700">{msg.text}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground text-right">{msg.ts}</p>
                  </div>
                </div>
              );
            }
            if (msg.kind === "loading") {
              return <LoadingBubble key={msg.id} label={msg.label} />;
            }
            if (msg.kind === "capture") {
              return <CaptureMessage key={msg.id} result={msg.result} ts={msg.ts} />;
            }
            if (msg.kind === "resolution") {
              return <ResolutionMessage key={msg.id} data={msg.data} original_b64={msg.original_b64} ts={msg.ts} />;
            }
            return null;
          })}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="shrink-0">
          <ChatInput
            onSend={handleSend}
            onCapture={handleCapture}
            disabled={busy || countdown !== null}
          />
        </div>
      </div>
    </>
  );
}
