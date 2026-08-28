import { useQuery } from '@tanstack/react-query'
import { getStats, getKafkaStatus } from '@/api/client'

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

      <KafkaStatusSection />
    </div>
  )
}

function KafkaStatusSection() {
  const { data: kafka } = useQuery({
    queryKey: ['kafka-status'],
    queryFn: getKafkaStatus,
    staleTime: 10_000,
    refetchInterval: 10_000,
  })

  if (!kafka || !kafka.configured) return null

  const cs = kafka.consumer?.stats ?? {}
  const hasConsumerActivity = (cs.ingested ?? 0) + (cs.updated ?? 0) + (cs.deleted ?? 0) + (cs.skipped_duplicate ?? 0) > 0

  return (
    <>
      <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }} />
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
        Kafka Sharing:
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
        <span style={{ color: 'var(--text-muted)' }}>Topic:</span>
        <span style={{ color: 'var(--text)', fontFamily: 'monospace', fontSize: 10 }}>{kafka.topic}</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
        <span style={{ color: 'var(--text-muted)' }}>Producer:</span>
        <span style={{ color: kafka.producer?.ready ? 'var(--success)' : 'var(--text-muted)' }}>
          {kafka.producer?.ready ? '● Ready' : kafka.producer?.started ? '○ Started' : '○ Off'}
        </span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
        <span style={{ color: 'var(--text-muted)' }}>Consumer:</span>
        <span style={{ color: kafka.consumer?.running ? 'var(--success)' : 'var(--text-muted)' }}>
          {kafka.consumer?.running ? '● Listening' : '○ Off'}
        </span>
      </div>

      {kafka.current_user?.username && (
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
          <span style={{ color: 'var(--text-muted)' }}>User:</span>
          <span style={{ color: kafka.current_user.allowed ? 'var(--success)' : 'var(--warning)' }}>
            {kafka.current_user.username}
            {kafka.current_user.allowed ? ' ✓' : ' (not allowed)'}
          </span>
        </div>
      )}

      {kafka.allowed_users.length > 0 && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
          Allowed: {kafka.allowed_users.map(u => u.username).join(', ')}
        </div>
      )}

      {hasConsumerActivity && (
        <>
          <div style={{ borderTop: '1px solid var(--border)', margin: '6px 0' }} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>Consumer stats:</div>
          {(cs.ingested ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--success)' }}>Ingested</span>
              <span style={{ color: 'var(--text)' }}>{cs.ingested}</span>
            </div>
          )}
          {(cs.updated ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--accent)' }}>Updated</span>
              <span style={{ color: 'var(--text)' }}>{cs.updated}</span>
            </div>
          )}
          {(cs.deleted ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--danger)' }}>Deleted (by admin)</span>
              <span style={{ color: 'var(--text)' }}>{cs.deleted}</span>
            </div>
          )}
          {(cs.skipped_duplicate ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--text-muted)' }}>Skipped (dup)</span>
              <span style={{ color: 'var(--text)' }}>{cs.skipped_duplicate}</span>
            </div>
          )}
          {(cs.skipped_unauthorized_delete ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--warning)' }}>Refused deletes</span>
              <span style={{ color: 'var(--text)' }}>{cs.skipped_unauthorized_delete}</span>
            </div>
          )}
          {(cs.errors ?? 0) > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 1 }}>
              <span style={{ color: 'var(--danger)' }}>Errors</span>
              <span style={{ color: 'var(--text)' }}>{cs.errors}</span>
            </div>
          )}
        </>
      )}
    </>
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
