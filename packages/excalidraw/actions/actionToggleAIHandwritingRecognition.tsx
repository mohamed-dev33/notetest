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
});
