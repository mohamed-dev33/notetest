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
});
