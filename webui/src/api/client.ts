// ── Typed fetch client for the WebUI REST API ────────────────────────────────
//
// All functions throw on non-2xx responses so callers can use try/catch
// or TanStack Query's error state directly.

import type {
  Memory,
  Stats,
  Config,
  VectorStats,
  VectorDetail,
  Peer,
  Identity,
  MigratePreviewItem,
  MigrateResult,
  CompareResult,
  SearchParams,
  MemoryCreate,
  MemoryUpdate,
  PgConnectionParams,
} from './types'

const BASE = '/api/v1'

// ── Helpers ───────────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? body.reason ?? detail
    } catch {
      // ignore parse errors
    }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json() as Promise<T>
}

function buildQuery(params: Record<string, unknown>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') {
      q.set(k, String(v))
    }
  }
  const s = q.toString()
  return s ? `?${s}` : ''
}

// ── Config / Identity / Peers ─────────────────────────────────────────────────

export const getConfig = (): Promise<Config> =>
  request('/config')

export const getIdentity = (): Promise<Identity> =>
  request('/identity')

export const getPeers = (): Promise<{ peers: Peer[] }> =>
  request('/peers')

// ── Stats ─────────────────────────────────────────────────────────────────────

export const getStats = (): Promise<{ success: boolean; data: Stats }> =>
  request('/stats')

// ── Memories ──────────────────────────────────────────────────────────────────

export const listMemories = (
  params: SearchParams = {},
): Promise<{ success: boolean; total: number; offset: number; limit: number; data: Memory[] }> =>
  request(`/memories${buildQuery(params as Record<string, unknown>)}`)

export const getMemory = (id: string): Promise<Memory> =>
  request(`/memories/${id}`)

export const createMemory = (
  body: MemoryCreate,
): Promise<{ success: boolean; data: Memory[] }> =>
  request('/memories', { method: 'POST', body: JSON.stringify(body) })

export const updateMemory = (
  id: string,
  body: MemoryUpdate,
): Promise<{ success: boolean; data: Array<{ id: string; updated: boolean }> }> =>
  request(`/memories/${id}`, { method: 'PATCH', body: JSON.stringify(body) })

export const deleteMemory = (
  id: string,
): Promise<{ success: boolean; data: Array<{ id: string; deleted: boolean }> }> =>
  request(`/memories/${id}`, { method: 'DELETE' })

// ── Backup / Export ───────────────────────────────────────────────────────────

export const createBackup = (): Promise<{
  success: boolean
  data: Array<{ backup_path: string; files_backed_up: string[] }>
}> => request('/backup', { method: 'POST' })

/** Triggers a file download by creating a temporary <a> element. */
export function downloadExport(): void {
  const a = document.createElement('a')
  a.href = BASE + '/export'
  a.download = ''
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// ── Vectors ───────────────────────────────────────────────────────────────────

export const getVectorStats = (): Promise<VectorStats> =>
  request('/vectors/stats')

export const getVector = (id: string): Promise<VectorDetail> =>
  request(`/vectors/${id}`)

export const rebuildVectors = (): Promise<{
  success: boolean
  data: Array<{ count: number }>
}> => request('/vectors/rebuild', { method: 'POST' })

// ── Migration ─────────────────────────────────────────────────────────────────

export const testPgConnection = (
  params: PgConnectionParams,
): Promise<{
  success: boolean
  has_memories_table?: boolean
  has_vectors_table?: boolean
  memory_count?: number
  vector_count?: number
  error?: string
}> =>
  request('/migrate/test-connection', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export interface MigratePreviewRequest {
  direction: string
  source_db_path?: string
  pg_host?: string
  pg_port?: number
  pg_database?: string
  pg_user?: string
  pg_password?: string
  limit?: number
}

export const previewMigration = (
  body: MigratePreviewRequest,
): Promise<{ success: boolean; count: number; memories: MigratePreviewItem[] }> =>
  request('/migrate/preview', { method: 'POST', body: JSON.stringify(body) })

export interface MigrateRequest {
  direction: string
  source_db_path?: string
  source_chroma_path?: string
  memory_ids?: string[]
  skip_duplicates?: boolean
  migrate_vectors?: boolean
  pg_host?: string
  pg_port?: number
  pg_database?: string
  pg_user?: string
  pg_password?: string
}

export const runMigration = (
  body: MigrateRequest,
): Promise<{ success: boolean; data: MigrateResult }> =>
  request('/migrate', { method: 'POST', body: JSON.stringify(body) })

// ── Compare ───────────────────────────────────────────────────────────────────

export const compareBackends = (
  params: PgConnectionParams,
): Promise<CompareResult> =>
  request(`/compare${buildQuery(params as unknown as Record<string, unknown>)}`)
