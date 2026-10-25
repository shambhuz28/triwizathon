import { useState } from 'react'
import type { AnalyzeResponse, LadderStep } from '../types'
import { Card, Pill, Stat, fmtInt } from '../ui'
import { BarChart, SubTabs } from './charts'
import { reductionRows } from './S6Panel'
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

type Tab = 'prompt' | 'charts' | 'checks'

export default function S7Panel({ r }: { r: AnalyzeResponse }) {
  const b = r.s7
  const [tab, setTab] = useState<Tab>('prompt')
  const [view, setView] = useState<'csv' | 'table' | 'plain'>(b.format === 'plain' ? 'plain' : 'csv')
  const [step, setStep] = useState<string | null>(null)
  const base = b.baseline
  const rows = parseCsv(b.csv)
  const shown = step ? b.steps.find(s => s.level === step) : null
  const fmtName = b.format === 'plain' ? 'plain text' : 'CSV'
  const problems = b.checks.missing_locks.length + b.checks.requirements_missing.length

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label={`Tokens (${fmtName})`} value={fmtInt(b.best_tokens)}
          sub={b.format === 'plain' ? `CSV would be ${fmtInt(b.csv_tokens)}` : `plain text would be ${fmtInt(b.tokens)}`} />
        <Stat label="vs baseline" value={`${b.best_tokens <= base.tokens ? '−' : '+'}${Math.abs((1 - b.best_tokens / Math.max(base.tokens, 1)) * 100).toFixed(0)}%`}
          sub={`${fmtInt(base.tokens)} · ${base.label.toLowerCase()}`} tone={b.best_tokens < base.tokens ? 'good' : 'default'} />
        <Stat label="Answer budget" value={fmtInt(b.answer_budget)} sub={`max_tokens ${b.max_tokens} · ~${b.answer_tokens_est} expected`} />
        <Stat label="Locks kept" value={b.checks.locks_preserved ? 'all' : `${b.checks.missing_locks.length} lost`}
          sub={`${r.s5.lock_set.length} locked items`} tone={b.checks.locks_preserved ? 'good' : 'bad'} />
      </div>

      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-900">
        <span className="font-semibold">Recommended format: {fmtName}</span>
        <span className="text-emerald-800">{b.format_reason}.</span>
        <button onClick={() => setTab('charts')} className="ml-auto text-xs font-medium text-emerald-700 underline">compare →</button>
      </div>

      <SubTabs value={tab} onChange={setTab} tabs={[
        { id: 'prompt', label: 'Prompt' },
        { id: 'charts', label: 'Format & token charts' },
        { id: 'checks', label: 'Checks & trace', badge: problems > 0 && <Pill className="bg-rose-100 text-rose-700">{problems}</Pill> },
      ]} />

      {tab === 'prompt' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2" title="Variant B · short, token-optimal"
            subtitle={`Ladder reached ${b.reached} (max ${b.max_ladder_level}) · every step checked; a failing step is undone`}
            right={<div className="flex items-center gap-2">
              <div className="flex rounded-md bg-slate-100 p-0.5 text-xs">
                {(['plain', 'csv', 'table'] as const).map(v => (
                  <button key={v} onClick={() => { setView(v); setStep(null) }}
                    className={`flex items-center gap-1 rounded px-2 py-1 font-medium ${view === v && !step ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
                    {v === 'csv' ? 'CSV' : v === 'table' ? 'CSV table' : 'Plain text'}
                    {(v === 'plain' ? 'plain' : v === 'csv' ? 'csv' : null) === b.format && <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" title="recommended" />}
                  </button>
                ))}
              </div>
              {view === 'plain' ? <CopyButtons text={b.prompt} filename="variant_b.txt" />
                : <CopyButtons text={b.csv} filename="variant_b.csv" type="text/csv" />}
            </div>}>
            <div className="max-h-[540px] overflow-auto rounded-lg bg-slate-50 p-4">
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
        </div>
      )}

      {tab === 'charts' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2" title="Tokens vs the original, step by step"
            subtitle="Input tokens after each ladder step that ran; the percentage is the change against the baseline">
            <BarChart labelWidth="w-64" caption="o200k tokenizer · L5 adds the answer-cap line, which pays for itself on the answer side"
              rows={reductionRows(r, [
                ...b.steps.filter(s => s.status === 'applied' || s.status === 'no change').map(s => ({
                  key: s.level, label: `${s.level} · ${s.name}`, value: s.tokens, tone: 'step' as const, tip: s.reason || s.status })),
                { key: 'best', label: `Variant B · ${fmtName} (send this)`, value: b.best_tokens, tone: 'accent' as const, tip: b.format_reason },
              ])} />
          </Card>
          <Card title="Plain text or CSV?" subtitle="Same content, two forms; the shorter one is sent">
            <BarChart labelWidth="w-24"
              rows={[
                { key: 'plain', label: 'Plain text', value: b.tokens, tone: b.format === 'plain' ? 'accent' : 'muted', note: b.format === 'plain' ? '· send' : undefined,
                  tip: 'Sentences joined in prompt order, data kept as written' },
                { key: 'csv', label: 'CSV', value: b.csv_tokens, tone: b.format === 'csv' ? 'accent' : 'muted', note: b.format === 'csv' ? '· send' : undefined,
                  tip: `${rows.length - 1} field,value rows; quotes inside values are doubled` },
              ]} caption="o200k tokenizer" />
            <p className="mt-3 text-xs text-slate-600">{b.format_reason}.</p>
            <p className="mt-1 text-[11px] text-slate-500">Other tokenizers can disagree: the Compare tab shows the winner per model and the measured Gemini and Ollama counts.</p>
          </Card>
          <ComparisonCard r={r} highlight="B" />
          <p className="-mt-2 text-[11px] text-slate-400 lg:col-span-3">Answer estimate: {b.answer_basis}.</p>
        </div>
      )}

      {tab === 'checks' && (
        <div className="grid gap-4 md:grid-cols-2">
          <ChecksCard checks={b.checks} />
          <TraceCard trace={b.trace} />
        </div>
      )}
    </div>
  )
}
