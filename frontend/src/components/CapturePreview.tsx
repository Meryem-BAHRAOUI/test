import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CapturePreviewProps {
  imageUrl: string;
  onDismiss: () => void;
}

const CapturePreview = ({ imageUrl, onDismiss }: CapturePreviewProps) => (
  <div className="animate-message-in rounded-2xl overflow-hidden border border-border bg-card shadow-md relative">
    <Button
      variant="ghost"
      size="icon"
      className="absolute top-2 right-2 h-7 w-7 rounded-full bg-card/80 backdrop-blur-sm hover:bg-card"
      onClick={onDismiss}
    >
      <X className="h-4 w-4" />
    </Button>
    <img
      src={imageUrl}
      alt="Screen capture"
      className="w-full h-48 object-cover"
    />
    <div className="px-3 py-2">
      <p className="text-xs text-muted-foreground">Screen capture preview</p>
    </div>
  </div>
);

export default CapturePreview;
