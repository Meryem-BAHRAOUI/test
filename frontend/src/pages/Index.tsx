import React, { useState, useRef, useCallback, useEffect } from "react";
import { Monitor, Download, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

const ANALYZER_URL = "http://127.0.0.1:8002";
const COUNTDOWN_SEC = 5;

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

// ── SVG overlay sur l'image originale ────────────────────────────────────────
function BboxOverlay({
  elements, ocr, imgW, imgH, displayW, displayH,
}: {
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
            <rect x={x1 * sx} y={y1 * sy} width={(x2 - x1) * sx} height={(y2 - y1) * sy}
              fill="none" stroke="#3b82f6" strokeWidth="2" />
            <text x={x1 * sx + 2} y={Math.max(y1 * sy - 3, 12)}
              fill="#3b82f6" fontSize="11" fontWeight="bold">
              {el.id} {el.score.toFixed(2)}
            </text>
          </g>
        );
      })}
      {ocr.map((bl) => {
        const [x1, y1, x2, y2] = bl.bbox;
        return (
          <g key={bl.id}>
            <rect x={x1 * sx} y={y1 * sy} width={(x2 - x1) * sx} height={(y2 - y1) * sy}
              fill="rgba(0,200,80,0.08)" stroke="#00c850" strokeWidth="1.5" />
            <text x={x1 * sx + 2} y={Math.max(y1 * sy - 3, 12)}
              fill="#00c850" fontSize="10">
              {bl.text.length > 20 ? bl.text.slice(0, 20) + "…" : bl.text}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Countdown overlay plein écran ─────────────────────────────────────────────
function CountdownOverlay({ seconds }: { seconds: number }) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/75">
      <div className="text-[10rem] font-black text-white leading-none drop-shadow-xl">
        {seconds}
      </div>
      <p className="text-white text-2xl mt-6 font-semibold">
        Switch to the window you want to capture
      </p>
      <p className="text-gray-400 text-base mt-2">
        Screenshot will be taken automatically
      </p>
    </div>
  );
}

// ── Page principale ───────────────────────────────────────────────────────────
export default function Index() {
  const { toast } = useToast();
  const [result, setResult]           = useState<AnalysisResult | null>(null);
  const [countdown, setCountdown]     = useState<number | null>(null);
  const [analyzing, setAnalyzing]     = useState(false);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [imgSize, setImgSize]         = useState({ w: 0, h: 0 });
  const [dispSize, setDispSize]       = useState({ w: 0, h: 0 });

  const imgRef       = useRef<HTMLImageElement>(null);
  const fetchStarted = useRef(false);

  // Countdown tick
  useEffect(() => {
    if (countdown === null) return;
    if (countdown === 0) {
      if (!fetchStarted.current) {
        fetchStarted.current = true;
        doCapture();
      }
      return;
    }
    const t = setTimeout(() => setCountdown((c) => (c ?? 1) - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  const handleCapture = () => {
    if (countdown !== null || analyzing) return;
    fetchStarted.current = false;
    setResult(null);
    setCountdown(COUNTDOWN_SEC);
  };

  const doCapture = async () => {
    setCountdown(null);
    setAnalyzing(true);
    try {
      const res = await fetch(`${ANALYZER_URL}/analyze/capture`, { method: "POST" });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data: AnalysisResult = await res.json();
      setResult(data);
      toast({
        title: "Analysis complete",
        description: `${data.ui_elements.length} UI elements · ${data.ocr_blocks.length} text blocks · Saved to data/`,
      });
    } catch (e: any) {
      toast({ title: "Capture failed", description: e.message, variant: "destructive" });
    } finally {
      setAnalyzing(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const clean = {
      screen_size:  result.screen_size,
      timestamp:    result.timestamp,
      ui_elements:  result.ui_elements,
      ocr_blocks:   result.ocr_blocks,
    };
    const blob = new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `analysis_${result.timestamp.replace(/[:.]/g, "-")}.json`;
    a.click();
  };

  const onImgLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight });
    setDispSize({ w: e.currentTarget.clientWidth,  h: e.currentTarget.clientHeight  });
  }, []);

  const isBusy = countdown !== null || analyzing;
  const currentSrc = result
    ? `data:image/png;base64,${showAnnotated ? result.image_b64 : result.original_b64}`
    : null;

  return (
    <>
      {countdown !== null && countdown > 0 && <CountdownOverlay seconds={countdown} />}

      <div className="min-h-screen bg-background p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Screen Analyzer</h1>
            <p className="text-muted-foreground text-sm mt-1">
              YOLOX UI detection + PaddleOCR text extraction
            </p>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline">Port 8002</Badge>
            <Badge>YOLOX-S</Badge>
            <Badge variant="secondary">PaddleOCR</Badge>
          </div>
        </div>

        {/* Capture card */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <p className="font-semibold">Capture Screen</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Click — you have <strong>5 seconds</strong> to switch to the window you want to capture.
                </p>
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <Button onClick={handleCapture} disabled={isBusy} className="min-w-[160px]">
                  <Monitor className="w-4 h-4 mr-2" />
                  {analyzing ? "Analyzing…" : countdown !== null ? `Switching… ${countdown}s` : "Capture Screen"}
                </Button>

                {result && (
                  <>
                    <Button variant="outline" onClick={() => setShowAnnotated((v) => !v)}>
                      <RefreshCw className="w-4 h-4 mr-2" />
                      {showAnnotated ? "Original" : "Annotated"}
                    </Button>
                    <Button variant="outline" onClick={downloadJson}>
                      <Download className="w-4 h-4 mr-2" />
                      JSON
                    </Button>
                  </>
                )}
              </div>
            </div>

            {result?.json_file && (
              <div className="mt-4 bg-green-50 border border-green-200 rounded-lg px-4 py-2 text-sm text-green-800 font-mono break-all">
                Saved → {result.json_file}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analyzing spinner */}
        {analyzing && (
          <Card>
            <CardContent className="pt-10 pb-10 text-center space-y-4">
              <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-muted-foreground font-medium">Running YOLOX + PaddleOCR…</p>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {result && !analyzing && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Image */}
            <div className="xl:col-span-2">
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">
                      {showAnnotated ? "Annotated" : "Original"} — {result.screen_size[0]}×{result.screen_size[1]}px
                    </CardTitle>
                    <span className="text-xs text-muted-foreground">{result.timestamp}</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="relative inline-block w-full">
                    <img
                      ref={imgRef}
                      src={currentSrc!}
                      alt="analyzed screenshot"
                      className="w-full rounded border"
                      onLoad={onImgLoad}
                    />
                    {!showAnnotated && (
                      <BboxOverlay
                        elements={result.ui_elements}
                        ocr={result.ocr_blocks}
                        imgW={imgSize.w} imgH={imgSize.h}
                        displayW={dispSize.w} displayH={dispSize.h}
                      />
                    )}
                  </div>
                  <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 border-2 border-blue-500 rounded" />
                      UI element (YOLOX)
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 border-2 border-green-500 rounded" />
                      Text block (OCR)
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right panel */}
            <div className="space-y-4">
              {/* Stats */}
              <Card>
                <CardHeader><CardTitle className="text-sm">Summary</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-blue-50 rounded-lg p-3 text-center">
                      <div className="text-3xl font-bold text-blue-600">{result.ui_elements.length}</div>
                      <div className="text-xs text-blue-500 mt-1">UI Elements</div>
                    </div>
                    <div className="bg-green-50 rounded-lg p-3 text-center">
                      <div className="text-3xl font-bold text-green-600">{result.ocr_blocks.length}</div>
                      <div className="text-xs text-green-500 mt-1">Text Blocks</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* UI Elements */}
              {result.ui_elements.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-sm flex items-center gap-2">
                    <span className="w-3 h-3 border-2 border-blue-500 rounded inline-block" />
                    UI Elements
                  </CardTitle></CardHeader>
                  <CardContent>
                    <div className="space-y-1 max-h-48 overflow-y-auto">
                      {result.ui_elements.map((el) => (
                        <div key={el.id} className="flex items-center justify-between text-sm py-1 border-b last:border-0">
                          <span className="font-mono text-blue-600 w-8">{el.id}</span>
                          <span className="text-muted-foreground text-xs flex-1 px-2">[{el.bbox.join(", ")}]</span>
                          <Badge variant={el.score >= 0.7 ? "default" : "secondary"}>
                            {(el.score * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* OCR blocks */}
              {result.ocr_blocks.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-sm flex items-center gap-2">
                    <span className="w-3 h-3 border-2 border-green-500 rounded inline-block" />
                    Text Blocks (OCR)
                  </CardTitle></CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {result.ocr_blocks.map((bl) => (
                        <div key={bl.id} className="text-sm pb-2 border-b last:border-0">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-green-600 text-xs">{bl.id}</span>
                            <Badge variant={bl.conf >= 0.8 ? "default" : "secondary"}>
                              {(bl.conf * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <div className="font-medium mt-0.5">{bl.text}</div>
                          <div className="text-muted-foreground text-xs">[{bl.bbox.join(", ")}]</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Raw JSON */}
              <Card>
                <CardHeader><CardTitle className="text-sm">JSON output</CardTitle></CardHeader>
                <CardContent>
                  <pre className="text-xs bg-gray-900 text-green-400 rounded p-3 overflow-auto max-h-72 leading-relaxed">
                    {JSON.stringify({
                      screen_size:  result.screen_size,
                      timestamp:    result.timestamp,
                      ui_elements:  result.ui_elements,
                      ocr_blocks:   result.ocr_blocks,
                    }, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!result && !isBusy && (
          <Card>
            <CardContent className="py-16 text-center">
              <Monitor className="w-14 h-14 text-muted-foreground/30 mx-auto mb-4" />
              <p className="font-medium text-muted-foreground">Click "Capture Screen" to start</p>
              <p className="text-sm text-muted-foreground/70 mt-1">
                You will have 5 seconds to switch to the target window.
                <br />
                Make sure <code className="bg-muted px-1 rounded">api_server.py</code> is running on port 8002.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
