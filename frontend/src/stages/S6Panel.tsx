import { useState } from 'react'
import type { AnalyzeResponse, SectionKey } from '../types'
import { Card, Pill, Stat, fmtInt } from '../ui'
import { BarChart, SubTabs } from './charts'
import type { BarRow } from './charts'
import { ChecksCard, ComparisonCard, CopyButtons, TraceCard } from './variants'

/** Baseline → after retrieval → after S5 → the variant, each with its change vs the baseline. */
export function reductionRows(r: AnalyzeResponse, last: BarRow[]): BarRow[] {
  const base = r.s6.baseline.tokens
  const pct = (v: number) => base ? `${v <= base ? '−' : '+'}${Math.abs((1 - v / base) * 100).toFixed(0)}%` : ''
  const rows: BarRow[] = [{ key: 'base', label: r.s6.baseline.label, value: base, tone: 'base', tip: 'What the variants are compared with' }]
  if (r.s0?.used) rows.push({ key: 'ret', label: 'After retrieval (S0)', value: r.s0.tokens_after, tone: 'step',
    tip: `${r.s0.passages.length} passage(s) instead of the whole files` })
  rows.push({ key: 's5', label: 'After S5 (unnecessary words removed)', value: r.s5.compression.compressed_tokens, tone: 'step',
    tip: r.s5.compression.method })
  return [...rows, ...last].map(row => row.key === 'base' ? row : { ...row, note: pct(row.value) })
}

const SECTION_STYLE: Record<SectionKey, string> = {
  role: 'border-violet-300', goal: 'border-blue-400', rules: 'border-amber-400', assumptions: 'border-orange-300',
  examples: 'border-teal-300', plan: 'border-sky-300', format: 'border-fuchsia-400', self_check: 'border-emerald-400',
  inputs: 'border-emerald-300', reminder: 'border-rose-300',
}
const SECTION_FILL: Record<SectionKey, string> = {
  role: 'bg-violet-400', goal: 'bg-blue-500', rules: 'bg-amber-500', assumptions: 'bg-orange-400', examples: 'bg-teal-500',
  plan: 'bg-sky-500', format: 'bg-fuchsia-500', self_check: 'bg-emerald-500', inputs: 'bg-emerald-300', reminder: 'bg-rose-400',
}

type View = 'prompt' | 'charts' | 'checks'

export default function S6Panel({ r }: { r: AnalyzeResponse }) {
  const a = r.s6
  const [tab, setTab] = useState<View>('prompt')
  const [view, setView] = useState<'prompt' | 'sections'>('prompt')
  const base = a.baseline
  const problems = a.checks.missing_locks.length + a.checks.requirements_missing.length + (a.checks.within_size_limit ? 0 : 1)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Tokens" value={fmtInt(a.tokens)} sub={`limit ${fmtInt(a.size_limit)}`} tone={a.checks.within_size_limit ? 'default' : 'bad'} />
        <Stat label="vs baseline" value={`${a.tokens <= base.tokens ? '−' : '+'}${Math.abs((1 - a.tokens / Math.max(base.tokens, 1)) * 100).toFixed(0)}%`}
          sub={`${fmtInt(base.tokens)} · ${base.label.toLowerCase()}`} tone={a.tokens <= base.tokens ? 'good' : 'default'} />
        <Stat label="Answer (est.)" value={`~${fmtInt(a.answer_tokens_est)}`} sub={`max_tokens ${a.max_tokens}`} />
        <Stat label="Locks kept" value={a.checks.locks_preserved ? 'all' : `${a.checks.missing_locks.length} lost`}
          sub={`${r.s5.lock_set.length} locked items`} tone={a.checks.locks_preserved ? 'good' : 'bad'} />
      </div>

      <SubTabs value={tab} onChange={setTab} tabs={[
        { id: 'prompt', label: 'Prompt' },
        { id: 'charts', label: 'Token charts' },
        { id: 'checks', label: 'Checks & trace', badge: problems > 0 && <Pill className="bg-rose-100 text-rose-700">{problems}</Pill> },
      ]} />

      {tab === 'prompt' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2" title="Variant A · detailed and precise"
            subtitle={`${a.sections.length} of 10 sections · static instructions first (cacheable), inputs fenced, key rules repeated at the end`}
            right={<div className="flex items-center gap-2">
              <div className="flex rounded-md bg-slate-100 p-0.5 text-xs">
                {(['prompt', 'sections'] as const).map(v => (
                  <button key={v} onClick={() => setView(v)}
                    className={`rounded px-2 py-1 font-medium ${view === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
                    {v === 'prompt' ? 'Prompt' : 'By section'}
                  </button>
                ))}
              </div>
              <CopyButtons text={a.prompt} filename="variant_a.txt" />
            </div>}>
            <div className="max-h-[600px] overflow-auto rounded-lg bg-slate-50 p-4">
              {view === 'prompt' ? <div className="prompt-text">{a.prompt}</div> : (
                <div className="space-y-3">
                  {a.sections.map(s => (
                    <div key={s.key} className={`border-l-4 pl-3 ${SECTION_STYLE[s.key]}`}>
                      <div className="mb-1 flex items-center gap-2 text-[11px]">
                        <span className="font-semibold uppercase tracking-wide text-slate-600">{s.title}</span>
                        <Pill className={s.source === 'copied' ? 'bg-slate-200 text-slate-700' : 'bg-indigo-100 text-indigo-700'}>{s.source}</Pill>
                        <span className="font-mono text-slate-400">{s.tokens} tok{s.segment_ids.length ? ` · ${s.segment_ids.join(' ')}` : ''}</span>
                      </div>
                      <div className="prompt-text text-[13px]">{s.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              Lines marked <span className="font-mono">[assumption: edit if needed]</span> fill a gap S3 found; nothing else is added beyond the prompt's own content and the constraints it implies.
            </p>
          </Card>
          <Card title="Sections left out" subtitle="A section appears only when the prompt supplies content or a gap needs flagging">
            {a.skipped.length === 0 ? <p className="text-sm text-slate-500">All ten sections are used.</p> : (
              <ul className="space-y-1.5 text-xs">
                {a.skipped.map(s => (
                  <li key={s.key}><span className="font-semibold text-slate-700">{s.title}</span> <span className="text-slate-500">— {s.reason}</span></li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}

      {tab === 'charts' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2" title="Tokens vs the original" subtitle="Input tokens at each step; the percentage is the change against the baseline">
            <BarChart labelWidth="w-56" caption={`o200k tokenizer · Variant A may grow: it adds format, rules and checks (limit ${fmtInt(a.size_limit)})`}
              rows={reductionRows(r, [{ key: 'A', label: 'Variant A · detailed', value: a.tokens, tone: 'accent', tip: `${a.sections.length} sections` }])} />
          </Card>
          <Card title="Where Variant A's tokens go" subtitle="Tokens per section">
            <div className="mb-2 flex h-3 overflow-hidden rounded" role="img" aria-label="Share of tokens per section">
              {a.sections.map(s => <div key={s.key} className={`${SECTION_FILL[s.key]} border-r-2 border-white last:border-r-0`} style={{ width: `${(s.tokens / Math.max(a.tokens, 1)) * 100}%` }} title={`${s.title} · ${s.tokens} tokens`} />)}
            </div>
            <ul className="space-y-1 text-xs">
              {a.sections.map(s => (
                <li key={s.key} className="flex items-center gap-2">
                  <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${SECTION_FILL[s.key]}`} />
                  <span className="flex-1 text-slate-700">{s.title}</span>
                  <span className="font-mono text-slate-500">{s.tokens} · {((s.tokens / Math.max(a.tokens, 1)) * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          </Card>
          <ComparisonCard r={r} highlight="A" />
          <p className="-mt-2 text-[11px] text-slate-400 lg:col-span-3">Answer estimate: {a.answer_basis}.</p>
        </div>
      )}

      {tab === 'checks' && (
        <div className="grid gap-4 md:grid-cols-2">
          <ChecksCard checks={a.checks} />
          <TraceCard trace={a.trace} />
        </div>
      )}
    </div>
  )
}
