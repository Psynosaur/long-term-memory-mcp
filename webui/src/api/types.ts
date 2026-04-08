// ── Types mirroring the FastAPI response models ──────────────────────────────

export interface Memory {
  id: string
  title: string
  content: string
  memory_type: string
  importance: number
  tags: string[]
  shared_with: string[]
  timestamp: string
  updated_at?: string
  last_accessed?: string
  token_count?: number
  reinforcement_accum?: number
  metadata?: Record<string, unknown>
  match_type?: string
  relevance_score?: number
  staleness_score?: number
  staleness_warning?: boolean
  // contradiction fields
  warning?: string
  conflicting_id?: string
  conflicting_title?: string
}

export interface Stats {
  total_memories: number
  memory_types: number
  avg_importance: number
  oldest_memory: string | null
  newest_memory: string | null
  type_breakdown: Record<string, number>
  type_token_breakdown: Record<string, { count: number; tokens: number }>
  database_backend: string
  vector_backend: string
  storage_size_mb: number
  db_size_mb: number
  vector_size_mb: number
  total_tokens: number
  avg_tokens: number
}

export interface Config {
  embedding_model: string
  embedding_model_config: {
    model_name: string
    dimensions: number
    max_tokens: number
    query_prefix: string
    description: string
  }
  database_backend: string
  vector_backend: string
  data_folder: string
}

export interface VectorStats {
  backend: string
  count: number
  embedding_model: string
  configured_dims: number
  actual_dims: number | null
  dims_match: boolean | null
  dims_warning: string | null
}

export interface VectorDetail {
  id: string
  document_preview: string | null
  metadata: Record<string, unknown> | null
  dimensions: number | null
  first_20: number[] | null
  stats: {
    dimensions: number
    min: number
    max: number
    mean: number
    l2_norm: number
  } | null
}

export interface Peer {
  node_uuid: string
  username: string
  host: string
  port: number
}

export interface Identity {
  node_uuid: string | null
  username: string | null
  created_at?: string
}

export interface MigratePreviewItem {
  id: string
  title: string
  memory_type: string
  importance: number
  tags: string[] | string
}

export interface MigrateResult {
  total_found: number
  migrated: number
  skipped_duplicates: number
  errors: number
  vectors_migrated: number
}

export interface CompareResult {
  sqlite_memory_count: number
  sqlite_vector_count: number
  pg_memory_count: number
  pg_vector_count: number
  only_in_sqlite: CompareRow[]
  only_in_pg: CompareRow[]
  modified: CompareRow[]
  identical: CompareRow[]
  summary: {
    only_sqlite: number
    only_pg: number
    modified: number
    identical: number
  }
}

export interface CompareRow {
  id: string
  title: string
  memory_type: string
  importance: number
  content_hash?: string
  hash_sqlite?: string
  hash_pg?: string
}

export interface SearchParams {
  q?: string
  type?: string
  min_importance?: number
  tags?: string
  date_from?: string
  date_to?: string
  sort?: string
  limit?: number
  offset?: number
  search_type?: 'structured' | 'semantic'
}

export interface MemoryCreate {
  title: string
  content: string
  memory_type?: string
  importance?: number
  tags?: string[]
  shared_with?: string[]
  file_paths?: string[]
}

export interface MemoryUpdate {
  title?: string
  content?: string
  memory_type?: string
  importance?: number
  tags?: string[]
  shared_with?: string[]
}

export interface PgConnectionParams {
  pg_host: string
  pg_port: number
  pg_database: string
  pg_user: string
  pg_password: string
}
