import { Loader2 } from "lucide-react";

const LoadingIndicator = () => (
  <div className="flex justify-start animate-message-in">
    <div className="flex items-center gap-2 rounded-2xl border border-border bg-chat-ai px-4 py-3 shadow-sm">
      <Loader2 className="h-4 w-4 animate-spin-slow text-muted-foreground" />
      <span className="text-sm text-muted-foreground">Thinking…</span>
    </div>
  </div>
);

export default LoadingIndicator;
