import React, { ReactNode } from "react";

interface PopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}

interface PopoverContentProps {
  className?: string;
  children: ReactNode;
}

interface PopoverTriggerProps {
  children: ReactNode;
  onClick?: () => void;
  asChild?: boolean;
}

export function Popover({ open, onOpenChange, children }: PopoverProps) {
  return (
    <div
      className="relative inline-block"
      onClick={(e) => {
        e.stopPropagation();
      }}
    >
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          if (child.type === PopoverTrigger) {
            return React.cloneElement(child, {
              onClick: () => onOpenChange(!open),
            } as any);
          }
          if (child.type === PopoverContent) {
            return open ? child : null;
          }
        }
        return child;
      })}
    </div>
  );
}

export function PopoverTrigger({ children, onClick, asChild }: PopoverTriggerProps) {
  // Popover injects onClick; attach it to the child element (asChild) or a
  // wrapping button, otherwise the popover can never open.
  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children as React.ReactElement<any>, { onClick });
  }
  return (
    <button type="button" onClick={onClick} className="inline-flex items-center">
      {children}
    </button>
  );
}

export function PopoverContent({ className = "", children }: PopoverContentProps) {
  return (
    <div
      className={`absolute top-full mt-2 z-50 w-72 rounded-md border bg-popover p-4 text-popover-foreground shadow-md ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {children}
    </div>
  );
}
