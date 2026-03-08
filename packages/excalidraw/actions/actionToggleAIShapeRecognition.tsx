import { CaptureUpdateAction } from "@excalidraw/element";

import { shapeRecognitionIcon } from "../components/icons";

import { register } from "./register";

export const actionToggleAIShapeRecognition = register({
  name: "aiShapeRecognition",
  icon: shapeRecognitionIcon,
  keywords: ["ai", "shape", "recognition", "detect"],
  label: "buttons.aiShapeRecognition",
  viewMode: false,
  trackEvent: {
    category: "canvas",
    predicate: (appState) => !appState.aiShapeRecognitionEnabled,
  },
  perform(elements, appState) {
    return {
      appState: {
        ...appState,
        aiShapeRecognitionEnabled: !this.checked!(appState),
      },
      captureUpdate: CaptureUpdateAction.EVENTUALLY,
    };
  },
  checked: (appState) => appState.aiShapeRecognitionEnabled,
  PanelComponent: ({ appState, updateData }) => (
    <button
      className={`ToolIcon_type_checkbox ${appState.aiShapeRecognitionEnabled ? "is-active" : ""}`}
      onClick={() => updateData(null)}
      title="AI Shape Recognition"
      style={{
        display: "flex",
        alignItems: "center",
        gap: "4px",
        padding: "4px 8px",
        borderRadius: "6px",
        border: "1px solid var(--default-border-color)",
        background: appState.aiShapeRecognitionEnabled
          ? "var(--color-primary)"
          : "var(--island-bg-color)",
        color: appState.aiShapeRecognitionEnabled
          ? "#fff"
          : "var(--text-primary-color)",
        cursor: "pointer",
        fontSize: "12px",
        whiteSpace: "nowrap",
      }}
    >
      🔷 AI Shape
    </button>
  ),
});
