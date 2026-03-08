/**
 * TrOCR Web Worker — runs handwriting recognition off the main thread.
 *
 * Loads the Xenova/trocr-small-handwritten model via @xenova/transformers
 * inside a Web Worker so inference never blocks the UI.
 *
 * Protocol:
 *   Main → Worker: { type: 'init' }
 *   Worker → Main: { type: 'ready' } | { type: 'error', message }
 *
 *   Main → Worker: { type: 'recognize', imageData: ImageData, id: number }
 *   Worker → Main: { type: 'result', text, confidence, id } | { type: 'error', message, id }
 */

let pipe: any = null;

async function loadModel() {
  const { pipeline, env } = await import("@xenova/transformers");
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  env.allowRemoteModels = true;
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
  }
  pipe = await pipeline("image-to-text", "Xenova/trocr-small-handwritten", {
    quantized: true,
  });
}

/**
 * Convert ImageData to a PNG blob URL using OffscreenCanvas.
 */
function imageDataToBlobUrl(imageData: ImageData): Promise<string> {
  const oc = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = oc.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);
  return oc.convertToBlob({ type: "image/png" }).then((blob) =>
    URL.createObjectURL(blob),
  );
}

self.onmessage = async (e: MessageEvent) => {
  const { type, id } = e.data;

  if (type === "init") {
    try {
      await loadModel();
      self.postMessage({ type: "ready" });
    } catch (err: any) {
      self.postMessage({ type: "error", message: err?.message ?? String(err) });
    }
    return;
  }

  if (type === "recognize") {
    if (!pipe) {
      self.postMessage({ type: "result", text: "", confidence: 0, id });
      return;
    }

    try {
      const imageData: ImageData = e.data.imageData;
      const url = await imageDataToBlobUrl(imageData);

      try {
        const results = await pipe(url);
        if (results && results.length > 0) {
          const text = (results[0].generated_text || "").trim();
          if (text.length > 0) {
            const hasAlphanumeric = /[a-zA-Z0-9]/.test(text);
            const confidence = hasAlphanumeric ? 85 : 40;
            self.postMessage({ type: "result", text, confidence, id });
            return;
          }
        }
      } finally {
        URL.revokeObjectURL(url);
      }

      self.postMessage({ type: "result", text: "", confidence: 0, id });
    } catch (err: any) {
      self.postMessage({
        type: "error",
        message: err?.message ?? String(err),
        id,
      });
    }
    return;
  }
};
