import { useState } from 'react'
import type { LockCategory, S5Result } from '../types'
import { Card, LOCK_STYLE, MarkedText, Pill } from '../ui'

export default function S5Panel({ s5, text }: { s5: S5Result; text: string }) {
  const cats = Object.keys(s5.by_category) as LockCategory[]
  const [hidden, setHidden] = useState<Set<LockCategory>>(new Set())
  const [showNested, setShowNested] = useState(false)
  const toggle = (c: LockCategory) => setHidden(h => { const n = new Set(h); if (n.has(c)) n.delete(c); else n.add(c); return n })

  const visible = s5.spans.filter(s => !hidden.has(s.category))
  // Show the innermost visible lock where locks nest, so numbers inside data still show.
  const marks = visible
    .filter(s => showNested ? !visible.some(o => o.parent === s.id) : s.parent === null || hidden.has(s5.spans.find(p => p.id === s.parent)!.category))
    .map(s => ({ start: s.start, end: s.end, key: s.id, className: `rounded px-0.5 ${LOCK_STYLE[s.category].bg}`, title: `${s.id} · ${s.category} · ${s.source}${s.detail ? ` · ${s.detail}` : ''}` }))

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="Locked words"
        subtitle="These must appear unchanged in every later version; S9 re-checks them. Click a category to hide it."
        right={<label className="flex items-center gap-1.5 text-xs text-slate-600"><input type="checkbox" checked={showNested} onChange={e => setShowNested(e.target.checked)} />show nested locks</label>}>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {cats.map(c => (
            <button key={c} onClick={() => toggle(c)}
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${hidden.has(c) ? 'bg-slate-100 text-slate-400 line-through' : LOCK_STYLE[c].bg}`}>
              <span className={`h-2 w-2 rounded-full ${LOCK_STYLE[c].dot}`} />{LOCK_STYLE[c].name} · {s5.by_category[c]}
            </button>
          ))}
        </div>
        <div className="max-h-[520px] overflow-auto rounded-lg bg-slate-50 p-4">
          <MarkedText text={text} marks={marks} />
        </div>
      </Card>

      <div className="space-y-4">
        <Card title="Allowed label sets" subtitle="Each label must survive exactly (no “Arts & Culture” → “Culture”)">
          {s5.label_sets.length === 0 ? <p className="text-sm text-slate-500">None found.</p> : s5.label_sets.map((ls, i) => (
            <div key={i} className="mb-2 flex flex-wrap gap-1.5">{ls.items.map(it => <Pill key={it} className="bg-yellow-100 text-yellow-800">{it}</Pill>)}</div>
          ))}
        </Card>

        <Card title={`Lock set · ${s5.lock_set.length}`} subtitle="Top-level strings handed to S6–S9 as untouchable">
          <ul className="max-h-[420px] space-y-1 overflow-auto">
            {s5.spans.filter(s => s.parent === null).map(s => (
              <li key={s.id} className="flex items-start gap-2 text-xs">
                <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${LOCK_STYLE[s.category].dot}`} />
                <span className="flex-1 break-all font-mono text-slate-700">{s.text.length > 80 ? s.text.slice(0, 80) + '…' : s.text}</span>
                <span className="shrink-0 text-[10px] text-slate-400">{s.source}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  )
}
