import { useEffect, useRef } from "react";

import "./AIRecognitionConfirm.scss";

interface AIRecognitionConfirmProps {
  type: "shape" | "text";
  label: string;
  confidence: number;
  position: { x: number; y: number };
  onAccept: () => void;
  onReject: () => void;
}

const AIRecognitionConfirm = ({
  type,
  label,
  confidence,
  position,
  onAccept,
  onReject,
}: AIRecognitionConfirmProps) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === "y" || e.key === "Y") {
        e.preventDefault();
        onAccept();
      } else if (e.key === "Escape" || e.key === "n" || e.key === "N") {
        e.preventDefault();
        onReject();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onAccept, onReject]);

  // Auto-dismiss after 8 seconds
  useEffect(() => {
    const timer = setTimeout(onReject, 8000);
    return () => clearTimeout(timer);
  }, [onReject]);

  const icon = type === "shape" ? "🔷" : "✍️";

  return (
    <div
      ref={ref}
      className="ai-recognition-confirm"
      style={{
        left: position.x,
        top: position.y,
      }}
    >
      <div className="ai-recognition-confirm__content">
        <span className="ai-recognition-confirm__icon">{icon}</span>
        <span className="ai-recognition-confirm__label">
          Replace with <strong>{label}</strong>?
          <span className="ai-recognition-confirm__confidence">
            {Math.round(confidence)}%
          </span>
        </span>
      </div>
      <div className="ai-recognition-confirm__actions">
        <button
          className="ai-recognition-confirm__btn ai-recognition-confirm__btn--accept"
          onClick={onAccept}
          title="Accept (Enter/Y)"
        >
          ✓
        </button>
        <button
          className="ai-recognition-confirm__btn ai-recognition-confirm__btn--reject"
          onClick={onReject}
          title="Reject (Esc/N)"
        >
          ✗
        </button>
      </div>
    </div>
  );
};

export default AIRecognitionConfirm;
