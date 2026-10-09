import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ToastProvider } from "@/lib/toast-context";
import { ToastDisplay } from "@/components/ui/toast-display";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <App />
        <ToastDisplay />
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>
);
