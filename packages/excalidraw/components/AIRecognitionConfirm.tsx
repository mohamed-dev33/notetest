import { useEffect, useRef } from "react";

import "./AIRecognitionConfirm.scss";

interface AIRecognitionConfirmProps {
  type: "shape";
  label: string;
  confidence: number;
  position: { x: number; y: number };
  alternatives?: { label: string; confidence: number; onAccept: () => void }[];
  onAccept: () => void;
  onReject: () => void;
}

const SHAPE_ICONS: Record<string, string> = {
  rectangle: "▭",
  ellipse: "⬭",
  diamond: "◇",
  triangle: "△",
  line: "─",
  arrow: "→",
};

const AIRecognitionConfirm = ({
  label,
  confidence,
  position,
  alternatives,
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
      } else if (alternatives && e.key >= "1" && e.key <= "9") {
        // Number keys select alternatives (1 = first alternative)
        const idx = parseInt(e.key, 10) - 1;
        if (idx < alternatives.length) {
          e.preventDefault();
          alternatives[idx].onAccept();
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onAccept, onReject, alternatives]);

  // Auto-dismiss after 10 seconds (longer to give time to read alternatives)
  useEffect(() => {
    const timer = setTimeout(onReject, 10000);
    return () => clearTimeout(timer);
  }, [onReject]);

  const icon = SHAPE_ICONS[label] ?? "🔷";

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
          <strong>{label}</strong>
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
        {alternatives && alternatives.length > 0 && (
          <>
            {alternatives.map((alt, idx) => (
              <button
                key={alt.label}
                className="ai-recognition-confirm__btn ai-recognition-confirm__btn--alt"
                onClick={alt.onAccept}
                title={`Use ${alt.label} instead (${idx + 1})`}
              >
                {SHAPE_ICONS[alt.label] ?? alt.label.charAt(0)}
              </button>
            ))}
          </>
        )}
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
