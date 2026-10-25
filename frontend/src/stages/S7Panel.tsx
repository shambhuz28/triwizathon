import { useState } from 'react'
import type { AnalyzeResponse, LadderStep } from '../types'
import { Card, Pill, Stat, fmtInt } from '../ui'
import { ChecksCard, ComparisonCard, CopyButtons, TraceCard } from './variants'

const STATUS_STYLE: Record<LadderStep['status'], string> = {
  applied: 'bg-emerald-100 text-emerald-700',
  'no change': 'bg-slate-100 text-slate-600',
  reverted: 'bg-rose-100 text-rose-700',
  skipped: 'bg-slate-100 text-slate-400',
}

/** Minimal CSV parser for display (quoted cells, doubled quotes, newlines inside quotes). */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [[]]
  let cell = '', quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++ }
      else if (c === '"') quoted = false
      else cell += c
    } else if (c === '"') quoted = true
    else if (c === ',') { rows[rows.length - 1].push(cell); cell = '' }
    else if (c === '\n') { rows[rows.length - 1].push(cell); rows.push([]); cell = '' }
    else cell += c
  }
  rows[rows.length - 1].push(cell)
  return rows
}

export default function S7Panel({ r }: { r: AnalyzeResponse }) {
  const b = r.s7
  const [view, setView] = useState<'csv' | 'table' | 'plain'>('csv')
  const [step, setStep] = useState<string | null>(null)
  const base = b.baseline
  const rows = parseCsv(b.csv)
  const shown = step ? b.steps.find(s => s.level === step) : null

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="Variant B · short, token-optimal (CSV)"
        subtitle={`Ladder reached ${b.reached} (max ${b.max_ladder_level}) · every step checked; a failing step is undone`}
        right={<div className="flex items-center gap-2">
          <div className="flex rounded-md bg-slate-100 p-0.5 text-xs">
            {(['csv', 'table', 'plain'] as const).map(v => (
              <button key={v} onClick={() => { setView(v); setStep(null) }}
                className={`rounded px-2 py-1 font-medium ${view === v && !step ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
                {v === 'csv' ? 'CSV' : v === 'table' ? 'Table' : 'Plain text'}
              </button>
            ))}
          </div>
          {view === 'plain' ? <CopyButtons text={b.prompt} filename="variant_b.txt" />
            : <CopyButtons text={b.csv} filename="variant_b.csv" type="text/csv" />}
        </div>}>
        <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Tokens (CSV)" value={fmtInt(b.csv_tokens)} sub={`${fmtInt(b.tokens)} as plain text`} />
          <Stat label="vs baseline" value={`${b.csv_tokens <= base.tokens ? '−' : '+'}${Math.abs((1 - b.csv_tokens / Math.max(base.tokens, 1)) * 100).toFixed(0)}%`}
            sub={`${fmtInt(base.tokens)} · ${base.label.toLowerCase()}`} tone={b.csv_tokens < base.tokens ? 'good' : 'default'} />
          <Stat label="Answer budget" value={fmtInt(b.answer_budget)} sub={`max_tokens ${b.max_tokens} · ~${b.answer_tokens_est} expected`} />
          <Stat label="Locks kept" value={b.checks.locks_preserved ? 'all' : `${b.checks.missing_locks.length} lost`}
            sub={`${r.s5.lock_set.length} locked items`} tone={b.checks.locks_preserved ? 'good' : 'bad'} />
        </div>
        <div className="max-h-[520px] overflow-auto rounded-lg bg-slate-50 p-4">
          {shown ? <><p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">After {shown.level} · {shown.name} · {shown.tokens} tokens</p><div className="prompt-text">{shown.prompt}</div></>
            : view === 'csv' ? <div className="prompt-text">{b.csv}</div>
            : view === 'plain' ? <div className="prompt-text">{b.prompt}</div>
            : (
              <table className="w-full text-left text-xs">
                <thead className="text-[10px] uppercase tracking-wide text-slate-500"><tr>{rows[0].map((h, i) => <th key={i} className="py-1 pr-3">{h}</th>)}</tr></thead>
                <tbody>
                  {rows.slice(1).map((row, i) => (
                    <tr key={i} className="border-t border-slate-200 align-top">
                      <td className="py-1.5 pr-3 font-mono font-semibold text-slate-600">{row[0]}</td>
                      <td className="prompt-text py-1.5 text-[12px]">{row.slice(1).join(',')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
      </Card>

      <div className="space-y-4">
        <Card title="Reduction ladder" subtitle="Safest change first; click a step to see the prompt after it">
          <ol className="space-y-1.5">
            {b.steps.map(s => (
              <li key={s.level}>
                <button onClick={() => setStep(step === s.level ? null : s.level)}
                  className={`w-full rounded-lg border px-2.5 py-1.5 text-left text-xs ${step === s.level ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-slate-500">{s.level}</span>
                    <span className="font-semibold text-slate-800">{s.name}</span>
                    <Pill className={`ml-auto ${STATUS_STYLE[s.status]}`}>{s.status}</Pill>
                    <span className="w-10 text-right font-mono text-slate-500">{s.tokens}</span>
                  </div>
                  {s.reason && <p className="mt-0.5 text-[11px] text-slate-500">{s.reason}</p>}
                </button>
              </li>
            ))}
          </ol>
        </Card>
        <ChecksCard checks={b.checks} />
        <TraceCard trace={b.trace} />
      </div>

      <ComparisonCard r={r} highlight="B" />
      <p className="-mt-2 text-[11px] text-slate-400 lg:col-span-3">Answer estimate: {b.answer_basis}. Ladder token counts are for the plain-text form; the CSV adds field names and quoting.</p>
    </div>
  )
}
