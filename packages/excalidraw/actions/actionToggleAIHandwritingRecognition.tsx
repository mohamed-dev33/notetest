import { CaptureUpdateAction } from "@excalidraw/element";

import { handwritingRecognitionIcon } from "../components/icons";

import { register } from "./register";

export const actionToggleAIHandwritingRecognition = register({
  name: "aiHandwritingRecognition",
  icon: handwritingRecognitionIcon,
  keywords: ["ai", "handwriting", "text", "ocr", "recognition"],
  label: "buttons.aiHandwritingRecognition",
  viewMode: false,
  trackEvent: {
    category: "canvas",
    predicate: (appState) => !appState.aiHandwritingRecognitionEnabled,
  },
  perform(elements, appState) {
    // Preload the handwriting engine when enabling
    if (!this.checked!(appState)) {
      import("../ai").then(({ preloadHandwritingEngine }) =>
        preloadHandwritingEngine(),
      );
    }

    return {
      appState: {
        ...appState,
        aiHandwritingRecognitionEnabled: !this.checked!(appState),
      },
      captureUpdate: CaptureUpdateAction.EVENTUALLY,
    };
  },
  checked: (appState) => appState.aiHandwritingRecognitionEnabled,
  PanelComponent: ({ appState, updateData }) => (
    <button
      className={`ToolIcon_type_checkbox ${appState.aiHandwritingRecognitionEnabled ? "is-active" : ""}`}
      onClick={() => updateData(null)}
      title="AI Handwriting Recognition"
      style={{
        display: "flex",
        alignItems: "center",
        gap: "4px",
        padding: "4px 8px",
        borderRadius: "6px",
        border: "1px solid var(--default-border-color)",
        background: appState.aiHandwritingRecognitionEnabled
          ? "var(--color-primary)"
          : "var(--island-bg-color)",
        color: appState.aiHandwritingRecognitionEnabled
          ? "#fff"
          : "var(--text-primary-color)",
        cursor: "pointer",
        fontSize: "12px",
        whiteSpace: "nowrap",
      }}
    >
      ✍️ AI Text
    </button>
  ),
});
