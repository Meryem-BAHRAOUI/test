import { cn } from "@/lib/utils";

interface ChatBubbleProps {
  content: string;
  sender: "user" | "ai";
  timestamp: string;
}

const ChatBubble = ({ content, sender, timestamp }: ChatBubbleProps) => {
  const isUser = sender === "user";

  return (
    <div
      className={cn(
        "flex animate-message-in",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-chat-user"
            : "bg-chat-ai border border-border"
        )}
      >
        <p className="text-sm leading-relaxed text-foreground">{content}</p>
        <p className="mt-1 text-[10px] text-muted-foreground text-right">
          {timestamp}
        </p>
      </div>
    </div>
  );
};

export default ChatBubble;
