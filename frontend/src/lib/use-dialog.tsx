import { useState } from "react";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

export interface DialogOptions {
  title: string;
  description?: string;
  okText?: string;
  cancelText?: string;
}

export interface AlertOptions extends DialogOptions {
  onConfirm: () => void;
}

export interface ConfirmOptions extends DialogOptions {
  onConfirm: () => void;
  onCancel?: () => void;
  destructive?: boolean;
}

/**
 * useAlert: shows an alert dialog and calls onConfirm when OK is clicked
 */
export function useAlert() {
  const [state, setState] = useState<{ open: boolean; options: AlertOptions | null }>({
    open: false,
    options: null,
  });

  const show = (options: AlertOptions) => {
    setState({ open: true, options });
  };

  const hide = () => {
    setState((s) => ({ ...s, open: false }));
  };

  const handleOk = async () => {
    if (state.options?.onConfirm) {
      await state.options.onConfirm();
    }
    hide();
  };

  return { show, state, hide, handleOk };
}

/**
 * useConfirm: shows a confirm dialog and calls onConfirm/onCancel based on user choice
 */
export function useConfirm() {
  const [confirmState, setConfirmState] = useState<{ open: boolean; options: ConfirmOptions | null }>({
    open: false,
    options: null,
  });
  const [confirming, setConfirming] = useState(false);

  const show = (options: ConfirmOptions) => {
    setConfirmState({ open: true, options });
  };

  const hide = () => {
    setConfirmState((s) => ({ ...s, open: false }));
  };

  const handleConfirm = async () => {
    setConfirming(true);
    try {
      if (confirmState.options?.onConfirm) {
        await confirmState.options.onConfirm();
      }
    } finally {
      setConfirming(false);
    }
    hide();
  };

  const handleCancel = async () => {
    if (confirmState.options?.onCancel) {
      await confirmState.options.onCancel();
    }
    hide();
  };

  return { show, confirmState, hide, handleConfirm, handleCancel, confirming };
}

/**
 * AlertDialog component to render via useAlert hook
 */
export function AlertDialog({
  open,
  options,
  onOk,
  onClose,
}: {
  open: boolean;
  options: AlertOptions | null;
  onOk: () => void;
  onClose: () => void;
}) {
  if (!options) return null;
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{options.title}</DialogTitle>
          {options.description && (
            <DialogDescription>{options.description}</DialogDescription>
          )}
        </DialogHeader>
        <DialogFooter>
          <Button onClick={onOk}>{options.okText ?? "OK"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/**
 * ConfirmDialog component to render via useConfirm hook
 */
export function ConfirmDialog({
  open,
  options,
  onConfirm,
  onCancel,
  confirming,
}: {
  open: boolean;
  options: ConfirmOptions | null;
  onConfirm: () => void;
  onCancel: () => void;
  confirming: boolean;
}) {
  if (!options) return null;
  return (
    <Dialog open={open} onOpenChange={onCancel}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{options.title}</DialogTitle>
          {options.description && (
            <DialogDescription>{options.description}</DialogDescription>
          )}
        </DialogHeader>
        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={onCancel} disabled={confirming}>
            {options.cancelText ?? "Cancel"}
          </Button>
          <Button
            onClick={onConfirm}
            disabled={confirming}
            variant={options.destructive ? "destructive" : "default"}
          >
            {options.okText ?? "Confirm"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
