import { useState, type FormEvent } from "react";
import { Send, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSend: (message: string) => void;
  onCapture: () => void;
  disabled?: boolean;
}

const ChatInput = ({ onSend, onCapture, disabled }: ChatInputProps) => {
  const [value, setValue] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center gap-2 border-t border-border bg-card px-4 py-3"
    >
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={onCapture}
        className="shrink-0 text-muted-foreground hover:text-foreground"
      >
        <Camera className="h-5 w-5" />
      </Button>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Type a message…"
        disabled={disabled}
        className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
      />
      <Button
        type="submit"
        size="icon"
        disabled={disabled || !value.trim()}
        className="shrink-0 rounded-xl"
      >
        <Send className="h-4 w-4" />
      </Button>
    </form>
  );
};

export default ChatInput;
