import type { CustomGesture } from '../ml/fewShot';

const KEY = 'air-canvas.custom-gestures.v1';


export function loadCustomGestures(): CustomGesture[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as CustomGesture[];
    return Array.isArray(parsed) ? parsed.filter((g) => Array.isArray(g.prototype) && typeof g.threshold === 'number') : [];
  } catch {
    return [];
  }
}

export function saveCustomGestures(gestures: CustomGesture[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(gestures));
  } catch {
  }
}
