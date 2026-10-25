import { useState } from 'react'
import type { AnalyzeResponse, SectionKey } from '../types'
import { Card, Pill, Stat, fmtInt } from '../ui'
import { ChecksCard, ComparisonCard, CopyButtons, TraceCard } from './variants'

const SECTION_STYLE: Record<SectionKey, string> = {
  role: 'border-violet-300', goal: 'border-blue-400', rules: 'border-amber-400', assumptions: 'border-orange-300',
  examples: 'border-teal-300', plan: 'border-sky-300', format: 'border-fuchsia-400', self_check: 'border-emerald-400',
  inputs: 'border-emerald-300', reminder: 'border-rose-300',
}

export default function S6Panel({ r }: { r: AnalyzeResponse }) {
  const a = r.s6
  const [view, setView] = useState<'prompt' | 'sections'>('prompt')
  const base = a.baseline

  return (
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
        <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Tokens" value={fmtInt(a.tokens)} sub={`limit ${fmtInt(a.size_limit)}`} tone={a.checks.within_size_limit ? 'default' : 'bad'} />
          <Stat label="vs baseline" value={`${a.tokens <= base.tokens ? '−' : '+'}${Math.abs((1 - a.tokens / Math.max(base.tokens, 1)) * 100).toFixed(0)}%`}
            sub={`${fmtInt(base.tokens)} · ${base.label.toLowerCase()}`} tone={a.tokens <= base.tokens ? 'good' : 'default'} />
          <Stat label="Answer (est.)" value={`~${fmtInt(a.answer_tokens_est)}`} sub={`max_tokens ${a.max_tokens}`} />
          <Stat label="Locks kept" value={a.checks.locks_preserved ? 'all' : `${a.checks.missing_locks.length} lost`}
            sub={`${r.s5.lock_set.length} locked items`} tone={a.checks.locks_preserved ? 'good' : 'bad'} />
        </div>
        <div className="max-h-[560px] overflow-auto rounded-lg bg-slate-50 p-4">
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

      <div className="space-y-4">
        <ChecksCard checks={a.checks} />
        <Card title="Sections left out" subtitle="A section appears only when the prompt supplies content or a gap needs flagging">
          {a.skipped.length === 0 ? <p className="text-sm text-slate-500">All ten sections are used.</p> : (
            <ul className="space-y-1.5 text-xs">
              {a.skipped.map(s => (
                <li key={s.key}><span className="font-semibold text-slate-700">{s.title}</span> <span className="text-slate-500">— {s.reason}</span></li>
              ))}
            </ul>
          )}
        </Card>
        <TraceCard trace={a.trace} />
      </div>

      <ComparisonCard r={r} highlight="A" />
      <p className="-mt-2 text-[11px] text-slate-400 lg:col-span-3">Answer estimate: {a.answer_basis}.</p>
    </div>
  )
}
