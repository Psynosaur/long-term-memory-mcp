import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getSettings, updateSettings, getKafkaStatus, kafkaReloadUsers, getConfig } from '@/api/client'
import type { Settings, SettingsUpdate, KafkaStatus } from '@/api/types'

export function SettingsTab() {
  const qc = useQueryClient()

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const { data: kafkaStatus } = useQuery({
    queryKey: ['kafka-status'],
    queryFn: getKafkaStatus,
    staleTime: 10_000,
  })

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
  })

  const updateMut = useMutation({
    mutationFn: (body: SettingsUpdate) => updateSettings(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['settings'] })
      qc.invalidateQueries({ queryKey: ['kafka-status'] })
    },
  })

  const reloadUsersMut = useMutation({
    mutationFn: kafkaReloadUsers,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kafka-status'] })
      qc.invalidateQueries({ queryKey: ['settings'] })
    },
  })

  if (isLoading || !settings) {
    return (
      <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading settings…</div>
    )
  }

  function toggle(key: keyof SettingsUpdate) {
    updateMut.mutate({ [key]: !settings![key as keyof Settings] })
  }

  return (
    <div style={{ padding: 24, maxWidth: 820, margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: 'var(--accent)' }}>Settings</h2>
      <p style={{ margin: '0 0 20px', fontSize: 12, color: 'var(--text-muted)' }}>
        Runtime settings — changes take effect immediately but do not persist across restarts.
      </p>

      {/* ── Identity ────────────────────────────────────────────── */}
      <Section title="Identity">
        <InfoRow label="Username" value={settings.username ?? '—'} mono />
        <InfoRow label="Node UUID" value={settings.node_uuid ?? '—'} mono />
      </Section>

      {/* ── System Info ─────────────────────────────────────────── */}
      {config && (
        <Section title="System">
          <InfoRow label="Database Backend" value={config.database_backend} />
          <InfoRow label="Vector Backend" value={config.vector_backend} />
          <InfoRow label="Embedding Model" value={config.embedding_model} />
          <InfoRow label="Dimensions" value={String(config.embedding_model_config?.dimensions ?? '—')} />
          <InfoRow label="Max Tokens" value={String(config.embedding_model_config?.max_tokens ?? '—')} />
          <InfoRow label="Data Folder" value={config.data_folder} mono />
        </Section>
      )}

      {/* ── Kafka Sharing ───────────────────────────────────────── */}
      <Section title="Kafka Sharing">
        {settings.kafka_sharing_active ? (
          <>
            <ToggleRow
              label="Auto-publish to Kafka"
              description="Automatically produce a Kafka event on every remember() and update_memory() call. Disable to use manual produce only (via the 📡 button)."
              checked={settings.auto_kafka_produce}
              onChange={() => toggle('auto_kafka_produce')}
              disabled={updateMut.isPending}
            />

            {kafkaStatus && <KafkaDetail kafkaStatus={kafkaStatus} onReloadUsers={() => reloadUsersMut.mutate()} reloading={reloadUsersMut.isPending} />}
          </>
        ) : (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 0' }}>
            Kafka sharing is not active. Start the server with <code>--kafka-sharing</code> and configure
            {' '}<code>.env</code> with your Kafka credentials.
          </div>
        )}
      </Section>

      {/* ── Network Sharing ─────────────────────────────────────── */}
      <Section title="Network Sharing (mDNS)">
        <InfoRow
          label="Status"
          value={settings.network_sharing_active ? '● Active' : '○ Not active'}
          color={settings.network_sharing_active ? 'var(--success)' : 'var(--text-muted)'}
        />
        {!settings.network_sharing_active && (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            Enable with <code>--network-sharing</code> flag.
          </div>
        )}
      </Section>

      {/* ── Memory Behaviour ────────────────────────────────────── */}
      <Section title="Memory Behaviour">
        <ToggleRow
          label="Importance Decay"
          description="Gradually reduce importance of unaccessed memories over time based on their type and age."
          checked={settings.decay_enabled}
          onChange={() => toggle('decay_enabled')}
          disabled={updateMut.isPending}
        />
        <ToggleRow
          label="Reinforcement"
          description="Boost importance when memories are retrieved by search, keeping frequently accessed memories strong."
          checked={settings.reinforcement_enabled}
          onChange={() => toggle('reinforcement_enabled')}
          disabled={updateMut.isPending}
        />
        <ToggleRow
          label="Staleness Scoring"
          description="Append a staleness_score (0–1) to search results. Memories past their expected lifetime for their type are flagged."
          checked={settings.staleness_enabled}
          onChange={() => toggle('staleness_enabled')}
          disabled={updateMut.isPending}
        />
        <ToggleRow
          label="Contradiction Detection"
          description="On remember(), check for semantically similar existing memories and warn about potential contradictions."
          checked={settings.contradiction_detection_enabled}
          onChange={() => toggle('contradiction_detection_enabled')}
          disabled={updateMut.isPending}
        />
      </Section>

      {/* ── Decay Configuration (read-only) ─────────────────────── */}
      <Section title="Decay Configuration" subtitle="(read-only — edit config.py to change)">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 4 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, fontWeight: 600 }}>Half-life (days)</div>
            {Object.entries(settings.decay_half_life_days).map(([type, days]) => (
              <InfoRow key={type} label={type} value={String(days)} />
            ))}
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, fontWeight: 600 }}>Min importance</div>
            {Object.entries(settings.decay_min_importance).map(([type, min]) => (
              <InfoRow key={type} label={type} value={String(min)} />
            ))}
          </div>
        </div>
        <InfoRow label="Protected tags" value={settings.decay_protect_tags.join(', ') || '—'} />
      </Section>

      {/* ── Staleness Configuration (read-only) ─────────────────── */}
      <Section title="Staleness Configuration" subtitle="(read-only)">
        <InfoRow label="Warn threshold" value={String(settings.staleness_warn_threshold)} />
        <InfoRow label="Warn types" value={settings.staleness_warn_types.join(', ')} />
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, marginBottom: 4, fontWeight: 600 }}>
          Expected lifetime (days)
        </div>
        {Object.entries(settings.staleness_expected_lifetime_days).map(([type, days]) => (
          <InfoRow key={type} label={type} value={String(days)} />
        ))}
      </Section>

      {/* ── Contradiction Configuration (read-only) ─────────────── */}
      <Section title="Contradiction Detection" subtitle="(read-only)">
        <InfoRow label="Similarity threshold" value={String(settings.contradiction_similarity_threshold)} />
        <InfoRow label="Check types" value={settings.contradiction_check_types.join(', ')} />
      </Section>
    </div>
  )
}

// ── Sub-components ──────────────────────────────────────────────────────────

function Section({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 6,
      padding: 14,
      marginBottom: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          {title}
        </div>
        {subtitle && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{subtitle}</span>}
      </div>
      {children}
    </div>
  )
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  label: string
  description: string
  checked: boolean
  onChange: () => void
  disabled?: boolean
}) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      gap: 12,
      padding: '8px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, lineHeight: 1.4 }}>{description}</div>
      </div>
      <label style={{
        position: 'relative',
        display: 'inline-block',
        width: 40,
        height: 22,
        flexShrink: 0,
        marginTop: 2,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      }}>
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          disabled={disabled}
          style={{ opacity: 0, width: 0, height: 0 }}
        />
        <span style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          background: checked ? 'var(--accent)' : 'var(--border)',
          borderRadius: 11,
          transition: 'background 0.2s',
        }}>
          <span style={{
            position: 'absolute',
            height: 16,
            width: 16,
            left: checked ? 20 : 3,
            bottom: 3,
            background: 'white',
            borderRadius: '50%',
            transition: 'left 0.2s',
          }} />
        </span>
      </label>
    </div>
  )
}

function InfoRow({
  label,
  value,
  mono,
  color,
}: {
  label: string
  value: string
  mono?: boolean
  color?: string
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}:</span>
      <span style={{
        color: color ?? 'var(--text)',
        fontFamily: mono ? 'monospace' : 'inherit',
        fontSize: mono ? 11 : 12,
        wordBreak: 'break-all',
        textAlign: 'right',
        maxWidth: '60%',
      }}>
        {value}
      </span>
    </div>
  )
}

function KafkaDetail({
  kafkaStatus,
  onReloadUsers,
  reloading,
}: {
  kafkaStatus: KafkaStatus
  onReloadUsers: () => void
  reloading: boolean
}) {
  const noAllowed = !kafkaStatus.allowed_users?.length

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
        <InfoRow label="Topic" value={kafkaStatus.topic ?? '—'} mono />
        <InfoRow
          label="Producer"
          value={kafkaStatus.producer?.ready ? '● Ready' : kafkaStatus.producer?.started ? '○ Started' : '○ Off'}
          color={kafkaStatus.producer?.ready ? 'var(--success)' : 'var(--text-muted)'}
        />
        <InfoRow
          label="Consumer"
          value={kafkaStatus.consumer?.running ? '● Listening' : '○ Off'}
          color={kafkaStatus.consumer?.running ? 'var(--success)' : 'var(--text-muted)'}
        />
        <InfoRow
          label="Mode"
          value={
            kafkaStatus.allowed_users.length > 0 && kafkaStatus.producer?.ready
              ? 'Producer + Consumer'
              : kafkaStatus.consumer?.running
                ? 'Consumer only (read-only)'
                : 'Off'
          }
        />
      </div>

      {/* Allowed users / producers list */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}>
            Allowed Producers
          </span>
          <button
            onClick={onReloadUsers}
            disabled={reloading}
            style={{ fontSize: 10, padding: '2px 8px' }}
          >
            {reloading ? 'Reloading…' : '↻ Reload from .env'}
          </button>
        </div>

        {noAllowed ? (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '4px 0' }}>
            No users configured — running in <strong>consumer-only</strong> mode.
            <br />
            Add entries to <code>ALLOWED_KAFKA_USERS</code> in <code>.env</code> to enable producing.
          </div>
        ) : (
          <div>
            {kafkaStatus.allowed_users.map((u) => (
              <div key={u.node_uuid} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2, padding: '2px 0' }}>
                <span style={{ color: 'var(--text)' }}>{u.username}</span>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace', fontSize: 10 }}>
                  {u.node_uuid.slice(0, 8)}…
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Consumer stats */}
      {kafkaStatus.consumer?.running && (() => {
        const cs = kafkaStatus.consumer.stats ?? {}
        const hasActivity = (cs.ingested ?? 0) + (cs.updated ?? 0) + (cs.deleted ?? 0) + (cs.skipped_duplicate ?? 0) > 0
        if (!hasActivity) return null
        return (
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>
              Consumer Activity
            </div>
            {(cs.ingested ?? 0) > 0 && <InfoRow label="Ingested" value={String(cs.ingested)} color="var(--success)" />}
            {(cs.updated ?? 0) > 0 && <InfoRow label="Updated" value={String(cs.updated)} color="var(--accent)" />}
            {(cs.deleted ?? 0) > 0 && <InfoRow label="Deleted" value={String(cs.deleted)} color="var(--danger)" />}
            {(cs.skipped_duplicate ?? 0) > 0 && <InfoRow label="Skipped (dup)" value={String(cs.skipped_duplicate)} />}
            {(cs.skipped_unauthorized_delete ?? 0) > 0 && (
              <InfoRow label="Refused deletes" value={String(cs.skipped_unauthorized_delete)} color="var(--warning)" />
            )}
            {(cs.errors ?? 0) > 0 && <InfoRow label="Errors" value={String(cs.errors)} color="var(--danger)" />}
          </div>
        )
      })()}
    </div>
  )
}
