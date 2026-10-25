import { useState } from 'react'
import type { ReactNode } from 'react'
import type { AnalyzeResponse, RequirementTrace, VariantCheck } from '../types'
import { Card, Pill, fmtInt, fmtUsd } from '../ui'

/** Copy + download buttons for a finished prompt. */
export function CopyButtons({ text, filename, type = 'text/plain' }: { text: string; filename: string; type?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type }))
    const a = Object.assign(document.createElement('a'), { href: url, download: filename })
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="flex items-center gap-1.5">
      <button onClick={download} className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-200">Download</button>
      <button onClick={copy} className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-indigo-700">{copied ? 'Copied' : 'Copy'}</button>
    </div>
  )
}

const pct = (v: number, base: number) => base ? `${v <= base ? '−' : '+'}${Math.abs((1 - v / base) * 100).toFixed(0)}%` : ''

/** Original (or whole files + prompt) vs Variant A vs Variant B, side by side. */
export function ComparisonCard({ r, highlight }: { r: AnalyzeResponse; highlight: 'A' | 'B' }) {
  const base = r.s6.baseline
  const retrieval = r.s0?.used
  const rows: { name: ReactNode; input: number; answer: number; perCall: number; per10k: number; key: string }[] = [
    { key: 'O', name: base.label, input: base.tokens, answer: base.cost.output_tokens_est, perCall: base.cost.per_call, per10k: base.cost.per_10k },
    ...(retrieval ? [{ key: 'R', name: 'After retrieval (S0 passages + prompt)', input: r.s0!.tokens_after, answer: r.s4.cost.output_tokens_est, perCall: r.s4.cost.per_call, per10k: r.s4.cost.per_10k }] : []),
    { key: 'A', name: 'Variant A · detailed', input: r.s6.tokens, answer: r.s6.answer_tokens_est, perCall: r.s6.cost.per_call, per10k: r.s6.cost.per_10k },
    { key: 'B', name: 'Variant B · short (CSV)', input: r.s7.csv_tokens, answer: r.s7.answer_tokens_est, perCall: r.s7.cost.per_call, per10k: r.s7.cost.per_10k },
  ]
  return (
    <Card className="lg:col-span-3" title="Before and after"
      subtitle={retrieval
        ? 'Document question: the baseline is every attached file in full plus the prompt — what it would take without retrieval'
        : `Input tokens counted with o200k; answer tokens are estimates (S3) · ${r.s4.cost.model_label}`}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-[10px] uppercase tracking-wide text-slate-500">
            <tr><th className="py-1.5 pr-3">Prompt</th><th className="pr-3 text-right">Input tokens</th><th className="pr-3 text-right">vs baseline</th>
              <th className="pr-3 text-right">Answer (est.)</th><th className="pr-3 text-right">Total</th><th className="pr-3 text-right">Per call</th><th className="text-right">Per 10k calls</th></tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.key} className={`border-t border-slate-100 ${row.key === highlight ? 'bg-indigo-50 font-semibold' : ''}`}>
                <td className="py-1.5 pr-3 text-slate-800">{row.name}</td>
                <td className="pr-3 text-right font-mono">{fmtInt(row.input)}</td>
                <td className={`pr-3 text-right font-mono ${row.key === 'O' ? 'text-slate-400' : row.input <= base.tokens ? 'text-emerald-600' : 'text-rose-600'}`}>{row.key === 'O' ? '—' : pct(row.input, base.tokens)}</td>
                <td className="pr-3 text-right font-mono">~{fmtInt(row.answer)}</td>
                <td className="pr-3 text-right font-mono">{fmtInt(row.input + row.answer)}</td>
                <td className="pr-3 text-right font-mono">{fmtUsd(row.perCall)}</td>
                <td className="text-right font-mono">{fmtUsd(row.per10k, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export function ChecksCard({ checks, extra }: { checks: VariantCheck; extra?: ReactNode }) {
  const item = (ok: boolean, label: string, detail?: string) => (
    <li className="flex items-start gap-2 text-xs">
      <span className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${ok ? 'bg-emerald-500' : 'bg-rose-500'}`}>{ok ? '✓' : '!'}</span>
      <span className="text-slate-700">{label}{detail && <span className="block text-slate-500">{detail}</span>}</span>
    </li>
  )
  return (
    <Card title="Checks" subtitle="Run on the finished prompt">
      <ul className="space-y-2">
        {item(checks.locks_preserved, 'Every locked word is present', checks.missing_locks.length ? `Missing: ${checks.missing_locks.join(' · ')}` : undefined)}
        {item(!checks.requirements_missing.length, `Requirements covered: ${checks.requirements_covered.length}`,
          checks.requirements_missing.length ? `Missing: ${checks.requirements_missing.join(' ')}` : undefined)}
        {item(checks.within_size_limit, 'Within the size limit')}
      </ul>
      {extra}
    </Card>
  )
}

export function TraceCard({ trace }: { trace: RequirementTrace[] }) {
  return (
    <Card title={`Where each requirement went · ${trace.length}`} subtitle="Every S2 requirement, traced into this variant">
      {trace.length === 0 ? <p className="text-sm text-slate-500">S2 found no requirements.</p> : (
        <ul className="space-y-1.5">
          {trace.map(t => (
            <li key={t.id} className="text-xs">
              <div className="flex items-center gap-2">
                <span className="font-mono font-semibold text-slate-500">{t.id}</span>
                <Pill className={t.where.startsWith('missing') ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'}>{t.where}</Pill>
              </div>
              <p className="mt-0.5 truncate text-slate-500" title={t.text}>{t.text}</p>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
