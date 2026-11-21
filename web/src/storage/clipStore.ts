export interface StoredClip {
  id?: number;
  label: string;
  frames: number[][];
  aspect: number;
  handedness: string | null;
  session: string;
  t: number;
}

const DB_NAME = 'air-canvas';
const STORE = 'clips';

let dbPromise: Promise<IDBDatabase | null> | null = null;
const memory: StoredClip[] = [];
let memoryId = 1;

function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }
  return dbPromise;
}

function request<T>(db: IDBDatabase, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = fn(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function addClip(clip: StoredClip): Promise<number> {
  const db = await openDb();
  if (!db) {
    const id = memoryId++;
    memory.push({ ...clip, id });
    return id;
  }
  return Number(await request(db, 'readwrite', (s) => s.add(clip)));
}

export async function allClips(): Promise<StoredClip[]> {
  const db = await openDb();
  if (!db) return [...memory];
  return request(db, 'readonly', (s) => s.getAll() as IDBRequest<StoredClip[]>);
}

export async function deleteClip(id: number): Promise<void> {
  const db = await openDb();
  if (!db) {
    const i = memory.findIndex((c) => c.id === id);
    if (i >= 0) memory.splice(i, 1);
    return;
  }
  await request(db, 'readwrite', (s) => s.delete(id));
}

export async function clearClips(): Promise<void> {
  const db = await openDb();
  if (!db) {
    memory.length = 0;
    return;
  }
  await request(db, 'readwrite', (s) => s.clear());
}


export function exportDocument(clips: StoredClip[], session: string): string {
  return JSON.stringify({
    format: 'air-canvas-clips',
    version: 1,
    fps: 30,
    frames_per_clip: clips[0]?.frames.length ?? 30,
    landmarks: 21,
    session,
    created: new Date().toISOString(),
    clips: clips.map(({ id: _id, ...c }) => c),
  });
}
