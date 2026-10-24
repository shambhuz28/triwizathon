import { useState } from 'react'
import type { LockCategory, PIRSegment, S5Result } from '../types'
import { Card, LABEL_STYLE, LOCK_STYLE, MarkedText, Pill, Stat, fmtInt } from '../ui'

const ACTION_STYLE: Record<PIRSegment['action'], string> = {
  keep: 'bg-slate-100 text-slate-600',
  compress: 'bg-indigo-100 text-indigo-700',
  drop: 'bg-rose-100 text-rose-700',
}

function CompressionCard({ s5, text }: { s5: S5Result; text: string }) {
  const c = s5.compression
  const [view, setView] = useState<'result' | 'removed'>('result')
  const [copied, setCopied] = useState(false)
  const lockText = Object.fromEntries(s5.spans.map(s => [s.id, s]))
  const removed = s5.segments.flatMap(p => p.dropped_spans.map(([start, end], i) => ({
    start, end, key: `${p.id}-${i}`,
    className: p.action === 'drop' ? 'text-slate-400 line-through decoration-slate-400' : 'rounded bg-rose-100 px-0.5 text-rose-700 line-through',
    title: `${p.id} · ${p.reason}`,
  })))
  const locks = s5.segments.flatMap(p => p.locks.map(id => lockText[id]).filter(Boolean).map(s => ({
    start: s.start, end: s.end, key: s.id, className: `rounded px-0.5 ${LOCK_STYLE[s.category].bg}`, title: `${s.id} · locked ${s.category}`,
  })))
  const copy = () => navigator.clipboard?.writeText(c.prompt).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })

  return (
    <Card className="lg:col-span-2" title="Prompt with unnecessary words removed" subtitle={c.method}
      right={<div className="flex items-center gap-2">
        <div className="flex rounded-md bg-slate-100 p-0.5 text-xs">
          {(['result', 'removed'] as const).map(v => (
            <button key={v} onClick={() => setView(v)}
              className={`rounded px-2 py-1 font-medium ${view === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
              {v === 'result' ? 'Result' : 'What was removed'}
            </button>
          ))}
        </div>
        <button onClick={copy} className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-indigo-700">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>}>
      <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Before" value={fmtInt(c.original_tokens)} sub={c.tokenizer} />
        <Stat label="After" value={fmtInt(c.compressed_tokens)} sub={`${(c.ratio * 100).toFixed(0)}% of the original`} />
        <Stat label="Saved" value={fmtInt(c.saved_tokens)} sub={`${((1 - c.ratio) * 100).toFixed(0)}% fewer tokens`} tone={c.saved_tokens > 0 ? 'good' : 'default'} />
        <Stat label="Locks kept" value={c.locks_preserved ? 'all' : `${c.missing_locks.length} lost`}
          sub={`${s5.lock_set.length} locked items`} tone={c.locks_preserved ? 'good' : 'bad'} />
      </div>
      {!c.locks_preserved && (
        <p className="mb-3 rounded-md bg-rose-50 p-2 text-xs text-rose-700">Missing locks: {c.missing_locks.join(' · ')}</p>
      )}
      <div className="max-h-[460px] overflow-auto rounded-lg bg-slate-50 p-4">
        {view === 'result'
          ? <div className="prompt-text">{c.prompt}</div>
          : <MarkedText text={text} marks={[...removed, ...locks]} />}
      </div>
      {view === 'removed' && (
        <p className="mt-2 text-[11px] text-slate-500">
          Grey strike-through: whole parts dropped (S2 padding, filler, repeats). Red strike-through: words removed by LLMLingua-2 + S4. Coloured: locked, always kept.
        </p>
      )}
    </Card>
  )
}

function HandoffCard({ s5 }: { s5: S5Result }) {
  const h = s5.handoff
  return (
    <Card title="Handoff to S6 / S7" subtitle="What Variant A and Variant B will be built from">
      <dl className="grid grid-cols-2 gap-2 text-xs">
        {[['Task', h.task_type], ['Difficulty', h.difficulty], ['Risk', h.risk_level], ['Max ladder level', h.max_ladder_level],
          ['Answer budget', `${h.answer_budget} tok`], ['max_tokens', `${h.max_tokens} tok`]].map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-50 px-2.5 py-2">
            <dt className="text-[10px] uppercase tracking-wide text-slate-500">{k}</dt>
            <dd className="font-semibold text-slate-800">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 space-y-1.5">
        {Object.entries(h.sections).map(([label, ids]) => (
          <div key={label} className="flex items-start gap-2 text-xs">
            <span className={`w-20 shrink-0 rounded px-1.5 py-0.5 text-center font-medium ${LABEL_STYLE[label as keyof typeof LABEL_STYLE]?.bg ?? 'bg-slate-100'}`}>{label}</span>
            <span className="font-mono text-slate-500">{ids.join(' ')}</span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-500">{h.requirements.length} requirement IDs to cover: <span className="font-mono">{h.requirements.join(' ')}</span></p>
      {h.assumptions.length > 0 && (
        <ul className="mt-3 space-y-1">
          {h.assumptions.map((a, i) => (
            <li key={i} className="rounded bg-amber-50 px-2 py-1 font-mono text-[11px] text-amber-800">{a.placeholder}</li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function PartsCard({ s5 }: { s5: S5Result }) {
  return (
    <Card className="lg:col-span-3" title={`Decision per part · ${s5.segments.length}`}
      subtitle="S2 label → keep, compress or drop; the keep rate is the floor S5 never goes below">
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-white text-[10px] uppercase tracking-wide text-slate-500">
            <tr><th className="py-1.5 pr-2">Part</th><th className="pr-2">Label</th><th className="pr-2">Action</th>
              <th className="pr-2 text-right">Floor</th><th className="pr-2 text-right">Tokens</th><th className="pr-2">Result</th><th>Why</th></tr>
          </thead>
          <tbody>
            {s5.segments.map(p => (
              <tr key={p.id} className="border-t border-slate-100 align-top">
                <td className="py-1.5 pr-2 font-mono text-slate-500">{p.requirement_id ?? p.id}</td>
                <td className="pr-2"><span className={`rounded px-1.5 py-0.5 font-medium ${LABEL_STYLE[p.label].bg}`}>{LABEL_STYLE[p.label].name}</span></td>
                <td className="pr-2"><Pill className={ACTION_STYLE[p.action]}>{p.action}</Pill></td>
                <td className="pr-2 text-right font-mono text-slate-500">{p.action === 'compress' ? `${(p.keep_rate * 100).toFixed(0)}%` : '—'}</td>
                <td className="whitespace-nowrap pr-2 text-right font-mono text-slate-500">{p.tokens} → {p.compressed_tokens}</td>
                <td className="max-w-md pr-2 font-mono text-slate-700">
                  {p.action === 'drop' ? <span className="text-slate-400 line-through">{p.text.slice(0, 90)}</span>
                    : (p.compressed.length > 140 ? p.compressed.slice(0, 140) + '…' : p.compressed)}
                </td>
                <td className="max-w-xs text-slate-500">{p.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

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
      <CompressionCard s5={s5} text={text} />
      <HandoffCard s5={s5} />
      <PartsCard s5={s5} />

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
