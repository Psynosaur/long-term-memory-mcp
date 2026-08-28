import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { compareBackends } from '@/api/client'
import type { CompareResult, CompareRow, PgConnectionParams } from '@/api/types'
import { Modal, Section } from './VectorsModal'

interface CompareModalProps {
  onClose: () => void
}

type Filter = 'only_sqlite' | 'only_pg' | 'modified' | 'identical'

export function CompareModal({ onClose }: CompareModalProps) {
  const [pg, setPg] = useState<PgConnectionParams>({
    pg_host: 'localhost', pg_port: 5433,
    pg_database: 'memories', pg_user: 'memory_user', pg_password: '',
  })
  const [result, setResult] = useState<CompareResult | null>(null)
  const [filters, setFilters] = useState<Set<Filter>>(new Set(['only_sqlite', 'only_pg', 'modified']))

  const compareMut = useMutation({
    mutationFn: () => compareBackends(pg),
    onSuccess: (res) => setResult(res),
  })

  function toggleFilter(f: Filter) {
    setFilters((prev) => {
      const next = new Set(prev)
      if (next.has(f)) next.delete(f)
      else next.add(f)
      return next
    })
  }

  const rows: Array<CompareRow & { _status: Filter }> = result
    ? [
        ...(filters.has('only_sqlite') ? result.only_in_sqlite.map((r) => ({ ...r, _status: 'only_sqlite' as Filter })) : []),
        ...(filters.has('only_pg') ? result.only_in_pg.map((r) => ({ ...r, _status: 'only_pg' as Filter })) : []),
        ...(filters.has('modified') ? result.modified.map((r) => ({ ...r, _status: 'modified' as Filter })) : []),
        ...(filters.has('identical') ? result.identical.map((r) => ({ ...r, _status: 'identical' as Filter })) : []),
      ]
    : []

  const STATUS_COLOR: Record<Filter, string> = {
    only_sqlite: 'var(--success)',
    only_pg: 'var(--info)',
    modified: 'var(--warning)',
    identical: 'var(--text-muted)',
  }
  const STATUS_LABEL: Record<Filter, string> = {
    only_sqlite: 'Only SQLite',
    only_pg: 'Only Postgres',
    modified: 'Modified',
    identical: 'Identical',
  }

  return (
    <Modal title="Database Comparison — SQLite vs Postgres" onClose={onClose} width={900}>
      {/* PG connection */}
      <Section title="PostgreSQL Connection">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', gap: 6 }}>
          {(['pg_host', 'pg_port', 'pg_database', 'pg_user', 'pg_password'] as (keyof PgConnectionParams)[]).map((k) => (
            <div key={k}>
              <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>{k.replace('pg_', '')}</label>
              <input
                type={k === 'pg_password' ? 'password' : 'text'}
                style={{ width: '100%' }}
                value={String(pg[k])}
                onChange={(e) => setPg({ ...pg, [k]: k === 'pg_port' ? Number(e.target.value) : e.target.value })}
              />
            </div>
          ))}
        </div>
        <button className="btn-accent" style={{ marginTop: 8 }} onClick={() => compareMut.mutate()} disabled={compareMut.isPending}>
          {compareMut.isPending ? 'Comparing…' : 'Compare Now'}
        </button>
        {compareMut.isError && (
          <span style={{ color: 'var(--danger)', fontSize: 12, marginLeft: 8 }}>
            {(compareMut.error as Error).message}
          </span>
        )}
      </Section>

      {result && (
        <>
          {/* Summary */}
          <div style={{ display: 'flex', gap: 16, fontSize: 12, marginBottom: 12, flexWrap: 'wrap' }}>
            <span>SQLite: <b>{result.sqlite_memory_count}</b> memories, <b>{result.sqlite_vector_count}</b> vectors</span>
            <span>Postgres: <b>{result.pg_memory_count}</b> memories, <b>{result.pg_vector_count}</b> vectors</span>
          </div>

          {/* Filter toggles */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
            {(['only_sqlite', 'only_pg', 'modified', 'identical'] as Filter[]).map((f) => {
              const count = result.summary[f === 'only_sqlite' ? 'only_sqlite' : f === 'only_pg' ? 'only_pg' : f]
              return (
                <label key={f} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 12 }}>
                  <input type="checkbox" checked={filters.has(f)} onChange={() => toggleFilter(f)} />
                  <span style={{ color: STATUS_COLOR[f] }}>{STATUS_LABEL[f]} ({count})</span>
                </label>
              )
            })}
          </div>

          {/* Diff table */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 4, overflow: 'auto', maxHeight: 380 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: 'var(--surface2)', position: 'sticky', top: 0 }}>
                  {['Status', 'ID', 'Title', 'Type', 'Imp'].map((h) => (
                    <th key={h} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '3px 8px', color: STATUS_COLOR[row._status], fontWeight: 600 }}>{STATUS_LABEL[row._status]}</td>
                    <td style={{ padding: '3px 8px', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>{row.id.slice(0, 15)}…</td>
                    <td style={{ padding: '3px 8px', maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.title}</td>
                    <td style={{ padding: '3px 8px' }}>{row.memory_type}</td>
                    <td style={{ padding: '3px 8px' }}>{row.importance}</td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr><td colSpan={5} style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)' }}>No rows match selected filters</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Modal>
  )
}
