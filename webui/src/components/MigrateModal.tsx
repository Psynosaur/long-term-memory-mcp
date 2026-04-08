import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { testPgConnection, previewMigration, runMigration } from '@/api/client'
import type { PgConnectionParams, MigratePreviewItem, MigrateResult } from '@/api/types'
import { Modal, Section } from './VectorsModal'

interface MigrateModalProps {
  onClose: () => void
}

type Direction = 'sqlite_to_sqlite' | 'sqlite_to_pg' | 'pg_to_sqlite'

export function MigrateModal({ onClose }: MigrateModalProps) {
  const [direction, setDirection] = useState<Direction>('sqlite_to_sqlite')
  const [sourceDbPath, setSourceDbPath] = useState('')
  const [skipDupes, setSkipDupes] = useState(true)
  const [migrateVectors, setMigrateVectors] = useState(true)
  const [pg, setPg] = useState<PgConnectionParams>({
    pg_host: 'localhost', pg_port: 5433,
    pg_database: 'memories', pg_user: 'memory_user', pg_password: '',
  })
  const [connStatus, setConnStatus] = useState<string | null>(null)
  const [preview, setPreview] = useState<MigratePreviewItem[] | null>(null)
  const [result, setResult] = useState<MigrateResult | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const testMut = useMutation({
    mutationFn: testPgConnection,
    onSuccess: (res) => {
      if (res.success) {
        setConnStatus(`Connected — ${res.memory_count} memories, ${res.vector_count} vectors`)
      } else {
        setConnStatus(`Failed: ${res.error}`)
      }
    },
    onError: (e: Error) => setConnStatus(`Error: ${e.message}`),
  })

  const previewMut = useMutation({
    mutationFn: previewMigration,
    onSuccess: (res) => setPreview(res.memories),
    onError: (e: Error) => setErr(e.message),
  })

  const migrateMut = useMutation({
    mutationFn: runMigration,
    onSuccess: (res) => setResult(res.data),
    onError: (e: Error) => setErr(e.message),
  })

  function handlePreview() {
    setErr(null)
    previewMut.mutate({
      direction,
      source_db_path: direction === 'sqlite_to_sqlite' ? sourceDbPath : undefined,
      ...pgFields(),
    })
  }

  function handleMigrate() {
    if (!window.confirm(`Run ${direction} migration?`)) return
    setErr(null)
    migrateMut.mutate({
      direction,
      source_db_path: direction === 'sqlite_to_sqlite' ? sourceDbPath : undefined,
      skip_duplicates: skipDupes,
      migrate_vectors: migrateVectors,
      ...pgFields(),
    })
  }

  function pgFields() {
    if (direction === 'sqlite_to_sqlite') return {}
    return {
      pg_host: pg.pg_host, pg_port: pg.pg_port,
      pg_database: pg.pg_database, pg_user: pg.pg_user, pg_password: pg.pg_password,
    }
  }

  return (
    <Modal title="Database Migration Tool" onClose={onClose} width={800}>
      {/* Direction */}
      <Section title="Migration Direction">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {(['sqlite_to_sqlite', 'sqlite_to_pg', 'pg_to_sqlite'] as Direction[]).map((d) => (
            <label key={d} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13 }}>
              <input type="radio" checked={direction === d} onChange={() => setDirection(d)} />
              {d === 'sqlite_to_sqlite' && 'SQLite file → Active SQLite (legacy import)'}
              {d === 'sqlite_to_pg' && 'Active SQLite/ChromaDB → pgvector/Postgres'}
              {d === 'pg_to_sqlite' && 'Postgres/pgvector → Active SQLite/ChromaDB'}
            </label>
          ))}
        </div>
      </Section>

      {/* Source path (sqlite_to_sqlite) */}
      {direction === 'sqlite_to_sqlite' && (
        <Section title="Source SQLite File">
          <input
            style={{ width: '100%' }}
            placeholder="/path/to/memories.db"
            value={sourceDbPath}
            onChange={(e) => setSourceDbPath(e.target.value)}
          />
        </Section>
      )}

      {/* PG connection */}
      {direction !== 'sqlite_to_sqlite' && (
        <Section title="PostgreSQL Connection">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', gap: 6 }}>
            <PgField label="Host" value={pg.pg_host} onChange={(v) => setPg({ ...pg, pg_host: v })} />
            <PgField label="Port" value={String(pg.pg_port)} onChange={(v) => setPg({ ...pg, pg_port: Number(v) })} />
            <PgField label="Database" value={pg.pg_database} onChange={(v) => setPg({ ...pg, pg_database: v })} />
            <PgField label="User" value={pg.pg_user} onChange={(v) => setPg({ ...pg, pg_user: v })} />
            <PgField label="Password" value={pg.pg_password} onChange={(v) => setPg({ ...pg, pg_password: v })} type="password" />
          </div>
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
            <button onClick={() => testMut.mutate(pg)} disabled={testMut.isPending}>
              {testMut.isPending ? 'Testing…' : 'Test Connection'}
            </button>
            {connStatus && <span style={{ fontSize: 12, color: connStatus.startsWith('Connected') ? 'var(--success)' : 'var(--danger)' }}>{connStatus}</span>}
          </div>
        </Section>
      )}

      {/* Options */}
      <Section title="Options">
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={skipDupes} onChange={(e) => setSkipDupes(e.target.checked)} />
          Skip duplicate memories (by content hash)
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input type="checkbox" checked={migrateVectors} onChange={(e) => setMigrateVectors(e.target.checked)} />
          Migrate vectors
        </label>
      </Section>

      {/* Preview */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button onClick={handlePreview} disabled={previewMut.isPending}>
          {previewMut.isPending ? 'Loading…' : 'Preview Source'}
        </button>
        <button className="btn-accent" onClick={handleMigrate} disabled={migrateMut.isPending}>
          {migrateMut.isPending ? 'Migrating…' : 'Start Migration'}
        </button>
      </div>

      {err && <div style={{ color: 'var(--danger)', fontSize: 12, marginBottom: 8 }}>{err}</div>}

      {result && (
        <div style={{ background: 'var(--bg)', border: '1px solid var(--success)', borderRadius: 4, padding: 10, fontSize: 12, marginBottom: 8 }}>
          <div style={{ color: 'var(--success)', fontWeight: 600, marginBottom: 4 }}>Migration complete</div>
          <div>Total found: {result.total_found}</div>
          <div>Migrated: {result.migrated}</div>
          <div>Skipped (duplicates): {result.skipped_duplicates}</div>
          <div>Errors: {result.errors}</div>
          <div>Vectors migrated: {result.vectors_migrated}</div>
        </div>
      )}

      {preview && preview.length > 0 && (
        <Section title={`Preview (${preview.length} memories)`}>
          <div style={{ maxHeight: 240, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 4 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: 'var(--surface2)' }}>
                  {['Title', 'Type', 'Imp'].map((h) => (
                    <th key={h} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((m) => (
                  <tr key={m.id} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '3px 8px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.title}</td>
                    <td style={{ padding: '3px 8px' }}>{m.memory_type}</td>
                    <td style={{ padding: '3px 8px' }}>{m.importance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </Modal>
  )
}

function PgField({ label, value, onChange, type = 'text' }: { label: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <div>
      <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>{label}</label>
      <input type={type} style={{ width: '100%' }} value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  )
}
