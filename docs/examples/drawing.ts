import { Skia } from "skia-wasm";
import type { Path } from "skia-wasm";

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice();
const element = document.querySelector("canvas")!;
const sk = await Skia.init({ device });
const surface = sk.surface(element);

const pen = sk.paint({
  color: "#0f172a",
  style: "stroke",
  strokeWidth: 4,
  strokeCap: "round",
  strokeJoin: "round",
});

// Closer points than this are hand tremor, not drawing.
const MIN_DISTANCE = 2;

// One path per stroke.
const strokes: Path[] = [];

// Nothing animates: a frame is drawn when a stroke changes, at most once per
// display frame however many events arrive.
const draw = () => {
  const canvas = surface.getCanvas();
  canvas.clear("white");
  for (const stroke of strokes) {
    canvas.drawPath(stroke, pen);
  }
  surface.flush();
};
let scheduled = false;
const requestDraw = () => {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      draw();
    });
  }
};
draw();

let current: Path | null = null;
// The last point taken in, not yet part of the path.
let last = { x: 0, y: 0 };

const position = (event: PointerEvent) => {
  const box = element.getBoundingClientRect();
  return { x: event.clientX - box.left, y: event.clientY - box.top };
};

element.addEventListener("pointerdown", (event) => {
  element.setPointerCapture(event.pointerId);
  last = position(event);
  current = sk.path().moveTo(last.x, last.y);
  strokes.push(current);
});

element.addEventListener("pointermove", (event) => {
  if (!current) {
    return;
  }
  // The browser sends one event per frame; the samples of the pointer in
  // between come with it, and a fast stroke would cut corners without them.
  const samples = event.getCoalescedEvents?.() ?? [event];
  for (const sample of samples.length > 0 ? samples : [event]) {
    const point = position(sample);
    if (Math.hypot(point.x - last.x, point.y - last.y) < MIN_DISTANCE) {
      continue;
    }
    // Each point pulls the curve towards it, and the curve ends halfway to
    // the next one: consecutive curves meet with the same tangent, so the
    // stroke has no corners.
    current.quadTo(
      last.x,
      last.y,
      (last.x + point.x) / 2,
      (last.y + point.y) / 2
    );
    last = point;
  }
  requestDraw();
});

const end = () => {
  if (current) {
    // The stroke reaches the last point. A tap is a line of no length,
    // which the round cap draws as a dot.
    current.lineTo(last.x, last.y);
    current = null;
    requestDraw();
  }
};
element.addEventListener("pointerup", end);
element.addEventListener("pointercancel", end);
