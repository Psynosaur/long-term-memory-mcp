import { useRef } from 'react'
import { useMemoryStore } from '@/store/memoryStore'
import { Search } from 'lucide-react'

const TYPES = ['', 'conversation', 'fact', 'preference', 'event', 'task', 'ephemeral']
const SORT_OPTIONS = [
  { value: 'importance DESC, timestamp DESC', label: 'Importance ↓ / Date ↓' },
  { value: 'timestamp DESC', label: 'Date ↓' },
  { value: 'timestamp ASC', label: 'Date ↑' },
  { value: 'importance DESC', label: 'Importance ↓' },
  { value: 'last_accessed DESC', label: 'Last Accessed ↓' },
  { value: 'last_accessed ASC', label: 'Last Accessed ↑' },
  { value: "json_extract(metadata, '$.reinforcement_accum') DESC, importance DESC", label: 'Reinforcement ↓' },
  { value: "json_extract(metadata, '$.reinforcement_accum') ASC, importance DESC", label: 'Reinforcement ↑' },
]

interface SearchPanelProps {
  onSearch: () => void
}

export function SearchPanel({ onSearch }: SearchPanelProps) {
  const {
    searchText, setSearchText,
    filterType, setFilterType,
    filterMinImportance, setFilterMinImportance,
    filterTags, setFilterTags,
    sortOrder, setSortOrder,
    searchMode, setSearchMode,
  } = useMemoryStore()

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function handleTextChange(v: string) {
    setSearchText(v)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(onSearch, 500)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') onSearch()
  }

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Search Memories
      </div>

      {/* Search text */}
      <div style={{ position: 'relative', marginBottom: 6 }}>
        <Search size={12} style={{ position: 'absolute', left: 7, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
        <input
          style={{ width: '100%', paddingLeft: 24, boxSizing: 'border-box' }}
          placeholder="Search…"
          value={searchText}
          onChange={(e) => handleTextChange(e.target.value)}
          onKeyDown={handleKeyDown}
        />
      </div>

      {/* Mode */}
      <div style={{ marginBottom: 6 }}>
        <label style={labelStyle}>Mode</label>
        <select
          value={searchMode}
          onChange={(e) => { setSearchMode(e.target.value as 'structured' | 'semantic'); onSearch() }}
          style={{ width: '100%' }}
        >
          <option value="structured">Text / Structured</option>
          <option value="semantic">Semantic (vector)</option>
        </select>
      </div>

      {/* Type + Min Importance — 2 column */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
        <div>
          <label style={labelStyle}>Type</label>
          <select
            value={filterType}
            onChange={(e) => { setFilterType(e.target.value); onSearch() }}
            style={{ width: '100%' }}
          >
            {TYPES.map((t) => (
              <option key={t} value={t}>{t || 'All'}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={labelStyle}>Min Imp.</label>
          <input
            type="number"
            min={1}
            max={10}
            value={filterMinImportance}
            onChange={(e) => setFilterMinImportance(Number(e.target.value))}
            onKeyDown={handleKeyDown}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Tags — full width */}
      <div style={{ marginBottom: 6 }}>
        <label style={labelStyle}>Tags (comma-separated)</label>
        <input
          placeholder="e.g. project, fact"
          value={filterTags}
          onChange={(e) => setFilterTags(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{ width: '100%', boxSizing: 'border-box' }}
        />
      </div>

      {/* Sort — full width */}
      <div style={{ marginBottom: 8 }}>
        <label style={labelStyle}>Sort</label>
        <select
          value={sortOrder}
          onChange={(e) => { setSortOrder(e.target.value); onSearch() }}
          style={{ width: '100%' }}
        >
          {SORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      <button className="btn-accent" style={{ width: '100%' }} onClick={onSearch}>
        Search
      </button>
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: 'var(--text-muted)',
  marginBottom: 3,
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}
