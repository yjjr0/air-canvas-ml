export const asset = (path: string): string => `${BASE}${path.replace(/^\//, '')}`;
export const AUTHOR = 'Yen Jun Jie Richard';
export const BASE = import.meta.env.BASE_URL;
export const BRUSH = { min: 0.004, max: 0.024, initial: 0.01, step: 0.004 };
export const MIN_FRAMES = 8; 
export const PEN_DOWN_BACKFILL_MS = 100; 
export const PEN_UP_TRIM_MS = 100;
export const REPO_URL = 'https://github.com/yjjr0/air-canvas-ml';
export const TRACK_LOST_MS = 250;

export type ActionId = keyof typeof ACTIONS;

export const ACTIONS = {
  none: 'Only show it',
  next_ink: 'Next ink',
  undo: 'Undo',
  redo: 'Redo',
  clear: 'Clear the canvas',
  brush_up: 'Thicker brush',
  brush_down: 'Thinner brush',
  save_png: 'Save as PNG',
} as const;

export type BaseLabel =
  | 'pinch_draw'
  | 'open_palm_hover'
  | 'fist_erase'
  | 'peace_color'
  | 'swipe_undo'
  | 'circle_clear'
  | 'idle';
export interface GestureInfo {
  name: string;
  does: string;
  how: string;
}

export const READOUT_ORDER: BaseLabel[] = [
  'pinch_draw',
  'open_palm_hover',
  'fist_erase',
  'peace_color',
  'swipe_undo',
  'circle_clear',
  'idle',
];

export const GESTURES: Record<BaseLabel, GestureInfo> = {
  pinch_draw: { name: 'Pinch', does: 'Draw', how: 'Touch your thumb and index fingertips, then move.' },
  open_palm_hover: { name: 'Open palm', does: 'Move without drawing', how: 'Spread your fingers.' },
  fist_erase: { name: 'Fist', does: 'Erase', how: 'Close your hand, then move.' },
  peace_color: { name: 'Peace sign', does: 'Next ink', how: 'Hold up index and middle fingers.' },
  swipe_undo: { name: 'Swipe', does: 'Undo', how: 'Flick an open hand sideways fast.' },
  circle_clear: { name: 'Circle', does: 'Clear the canvas', how: 'Trace one quick loop with your index finger.' },
  idle: { name: 'Anything else', does: 'Nothing', how: 'Other gestures are ignored.' },
};

export const COOLDOWN_MS: Record<string, number> = {
  peace_color: 700,
  swipe_undo: 800,
  circle_clear: 1500,
  custom: 1000,
};

export const FILTER = {
  enterThreshold: 0.6,
  exitThreshold: 0.4,
  defaultEnterFrames: 3,
  enterFrames: { pinch_draw: 2, circle_clear: 4 } as Record<string, number>,
  exitFrames: 2,
};

export const INKS = [
  { id: 'neon', name: 'Neon', color: '#ff5fcf' },
  { id: 'mercury', name: 'Mercury', color: '#4fe0ff' },
  { id: 'sodium', name: 'Sodium', color: '#ffb547' },
  { id: 'phosphor', name: 'Phosphor', color: '#a6ff5e' },
  { id: 'argon', name: 'Argon', color: '#9a8cff' },
  { id: 'magnesium', name: 'Magnesium', color: '#fff3e3' },
] as const;
