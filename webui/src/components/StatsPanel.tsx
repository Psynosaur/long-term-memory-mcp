import { useQuery } from '@tanstack/react-query'
import { getStats } from '@/api/client'

export function StatsPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
    refetchInterval: 30_000,
  })

  const stats = data?.data

  if (isLoading || !stats) {
    return (
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 12 }}>
        <SectionTitle>Statistics</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading…</div>
      </div>
    )
  }

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 12 }}>
      <SectionTitle>Statistics</SectionTitle>

      <StatRow label="Total Memories" value={stats.total_memories.toLocaleString()} />
      <StatRow label="Memory Types" value={String(stats.memory_types)} />
      <StatRow label="Avg Importance" value={stats.avg_importance.toFixed(1)} />
      <StatRow label="Total Tokens" value={stats.total_tokens.toLocaleString()} />
      <StatRow label="Avg Tokens" value={Math.round(stats.avg_tokens).toLocaleString()} />

      <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }} />

      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
        Embedding: <span style={{ color: 'var(--text)' }}>{stats.database_backend !== 'unknown' ? `bge-small (${stats.vector_backend})` : '—'}</span>
      </div>

      <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }} />

      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Breakdown:</div>
      {stats.type_token_breakdown
        ? Object.entries(stats.type_token_breakdown).map(([type, info]) => (
            <div key={type} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
              <span style={{ color: 'var(--text)' }}>{type}</span>
              <span style={{ color: 'var(--text-muted)' }}>
                {info.count} ({info.tokens.toLocaleString()} tokens)
              </span>
            </div>
          ))
        : Object.entries(stats.type_breakdown ?? {}).map(([type, count]) => (
            <div key={type} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
              <span style={{ color: 'var(--text)' }}>{type}</span>
              <span style={{ color: 'var(--text-muted)' }}>{count}</span>
            </div>
          ))}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
      {children}
    </div>
  )
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}:</span>
      <span style={{ color: 'var(--text)', fontWeight: 500 }}>{value}</span>
    </div>
  )
}
