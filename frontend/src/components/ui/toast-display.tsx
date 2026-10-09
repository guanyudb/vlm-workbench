import { useToast } from "@/lib/toast-context";
import { X, AlertCircle, CheckCircle2 } from "lucide-react";

export function ToastDisplay() {
  const { toasts, removeToast } = useToast();

  if (!toasts.length) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="pointer-events-auto flex items-start gap-3 rounded-lg border px-4 py-3 text-sm shadow-lg animate-in fade-in slide-in-from-right-4 duration-300"
          style={{
            backgroundColor: "var(--background)",
            borderColor: "var(--border)",
            ...(toast.variant === "destructive" && {
              backgroundColor: "hsl(0 84.2% 60.2% / 0.1)",
              borderColor: "hsl(0 84.2% 60.2%)",
            }),
            ...(toast.variant === "success" && {
              backgroundColor: "hsl(142.3 76.2% 36.3% / 0.1)",
              borderColor: "hsl(142.3 76.2% 36.3%)",
            }),
          }}
        >
          <div className="flex items-start gap-3 flex-1 pt-0.5">
            {toast.variant === "destructive" && (
              <AlertCircle className="size-4 shrink-0 text-destructive" />
            )}
            {toast.variant === "success" && (
              <CheckCircle2 className="size-4 shrink-0 text-emerald-600" />
            )}
            <div className="space-y-1">
              {toast.title && <div className="font-medium">{toast.title}</div>}
              {toast.description && (
                <div
                  className="text-xs text-muted-foreground"
                  style={{
                    ...(toast.variant === "destructive" && { color: "hsl(0 84.2% 60.2%)" }),
                    ...(toast.variant === "success" && { color: "hsl(142.3 76.2% 36.3%)" }),
                  }}
                >
                  {toast.description}
                </div>
              )}
            </div>
          </div>
          <button
            onClick={() => removeToast(toast.id)}
            className="shrink-0 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="size-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
