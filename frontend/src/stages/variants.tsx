import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { AnalyzeResponse, MarketModel, MarketResponse, MarketText, PromptKey, RequirementTrace, VariantCheck } from '../types'
import { Card, Pill, fmtInt, fmtUsd } from '../ui'
import { BarChart, PairChart, SeverityMeter } from './charts'

const OPTION_TONE: Record<PromptKey, 'base' | 'accent' | 'step'> = { original: 'base', A: 'accent', B: 'step' }

/** Which prompt to send, why, and what each choice trades away. */
export function RecommendationCard({ r }: { r: AnalyzeResponse }) {
  const rec = r.recommendation
  const name = (k: PromptKey) => rec.options.find(o => o.key === k)!.name
  return (
    <Card className="lg:col-span-3" title="Recommendation" subtitle="Chosen from the task's severity (S3) and each variant's checks; trade-offs come from the measured numbers">
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-3">
          <SeverityMeter level={rec.severity} score={rec.severity_score} max={rec.severity_max} />
          <p className="text-[11px] text-slate-500">{rec.factors.join(' · ')}</p>
          <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-3">
            <div className="text-[11px] font-medium uppercase tracking-wide text-indigo-700">Send</div>
            <div className="text-lg font-semibold text-slate-900">{name(rec.choice)}</div>
            {rec.fallback && <div className="text-xs text-slate-600">fallback: {name(rec.fallback)}</div>}
            <p className="mt-1.5 text-xs text-slate-700">{rec.reason}</p>
          </div>
          {rec.tips.length > 0 && <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-600">{rec.tips.map(t => <li key={t}>{t}</li>)}</ul>}
        </div>
        <div className="lg:col-span-2">
          <p className="mb-2 text-xs font-medium text-slate-700">Cost per month at {fmtInt(r.s4.cost.monthly_volume)} calls</p>
          <BarChart labelWidth="w-48" format={v => fmtUsd(v, 2)} unit="USD"
            rows={rec.options.map(o => ({
              key: o.key, label: o.name, value: o.per_month, tone: OPTION_TONE[o.key],
              note: o.key === rec.choice ? '· recommended' : o.key === rec.fallback ? '· fallback' : undefined,
              tip: <>{fmtInt(o.input_tokens)} input + ~{fmtInt(o.answer_tokens)} answer tokens · {fmtUsd(o.per_call)} per call</>,
            }))} />
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            {rec.options.map(o => (
              <div key={o.key} className={`rounded-lg border p-3 text-xs ${o.key === rec.choice ? 'border-indigo-400 ring-1 ring-indigo-400' : 'border-slate-200'}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-900">{o.name}</span>
                  {!o.checks_passed && <Pill className="bg-rose-100 text-rose-700">checks failed</Pill>}
                </div>
                <div className="mt-0.5 text-[11px] text-slate-500">Expected precision: <b className="text-slate-700">{o.precision}</b></div>
                <div className="text-[11px] text-slate-500">Best for {o.best_for}</div>
                <ul className="mt-2 space-y-1">
                  {o.pros.map(p => <li key={p} className="flex gap-1.5 text-slate-700"><span className="font-bold text-emerald-600" aria-label="gain">+</span>{p}</li>)}
                  {o.cons.map(p => <li key={p} className="flex gap-1.5 text-slate-700"><span className="font-bold text-rose-600" aria-label="cost">−</span>{p}</li>)}
                </ul>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-slate-400">{rec.note}</p>
        </div>
      </div>
    </Card>
  )
}

const TEXT_NAME: Record<MarketText, string> = {
  original: 'Original', retrieved: 'After retrieval', variant_a: 'Variant A', variant_b: 'B plain', variant_b_csv: 'B CSV',
}

/** Which form of Variant B this model's tokenizer counts shorter. */
function shorterForm(m: MarketModel): 'plain' | 'csv' | 'tie' | null {
  const p = m.tokens?.variant_b, c = m.tokens?.variant_b_csv
  return p == null || c == null ? null : p < c ? 'plain' : c < p ? 'csv' : 'tie'
}

/** Token count and cost of each prompt on every model on sale, from each vendor's own counter. */
export function MarketCard({ market, loading, error, onRetry, defaultFocus = 'variant_a' }: {
  market: MarketResponse | null; loading: boolean; error: string | null; onRetry: () => void; defaultFocus?: MarketText
}) {
  const [focus, setFocus] = useState<MarketText>(defaultFocus)
  const bothForms = !!market?.texts.includes('variant_b_csv')
  const formTally = useMemo(() => {
    const t = { plain: 0, csv: 0, tie: 0 }
    for (const m of market?.models ?? []) { const f = shorterForm(m); if (f) t[f]++ }
    return t
  }, [market])
  const [query, setQuery] = useState('')
  const [countedOnly, setCountedOnly] = useState(true)
  const [sort, setSort] = useState<'cost' | 'tokens' | 'name'>('cost')
  const [all, setAll] = useState(false)
  const rows = useMemo(() => {
    if (!market) return []
    const q = query.toLowerCase()
    const list = market.models.filter(m => (!countedOnly || m.tokens) && (!q || `${m.id} ${m.name}`.toLowerCase().includes(q)))
    const key = (m: MarketModel) => sort === 'name' ? 0 : sort === 'tokens' ? m.tokens?.[focus] ?? Infinity : m.cost?.[focus] ?? Infinity
    return sort === 'name' ? [...list].sort((a, b) => a.id.localeCompare(b.id)) : [...list].sort((a, b) => key(a) - key(b))
  }, [market, query, countedOnly, sort, focus])
  // chart: the newest counted model of each maker (the catalogue lists newest first)
  const flagships = useMemo(() => {
    if (!market) return []
    const seen = new Map<string, MarketModel>()
    for (const m of market.models) if (m.tokens && !seen.has(m.vendor)) seen.set(m.vendor, m)
    const counts = new Map<string, number>()
    for (const m of market.models) counts.set(m.vendor, (counts.get(m.vendor) ?? 0) + 1)
    return [...seen.values()].sort((a, b) => (counts.get(b.vendor) ?? 0) - (counts.get(a.vendor) ?? 0)).slice(0, 14)
  }, [market])

  return (
    <Card className="lg:col-span-3" title="Tokens on every model on the market"
      subtitle={market
        ? <>Counted with each vendor's own tokenizer or count-tokens API ({market.counted} of {market.total} models) · prices live from <a className="underline" href={market.source} target="_blank" rel="noreferrer">the OpenRouter catalogue</a>, {market.fetched_at} · nothing estimated</>
        : 'Each vendor counts with its own tokenizer; prices come from the live catalogue'}
      right={market && <span className="font-mono text-[11px] text-slate-500">o200k reference: {Object.entries(market.reference).map(([k, v]) => `${TEXT_NAME[k as MarketText]} ${fmtInt(v)}`).join(' · ')}</span>}>
      {loading && !market && <p className="text-sm text-slate-500">Fetching the catalogue and counting with each vendor's tokenizer… the first run downloads the open-weight tokenizers once (about half a minute).</p>}
      {error && <p className="text-sm text-rose-700">Could not count: {error}. <button className="underline" onClick={onRetry}>Retry</button></p>}
      {market && (
        <div className="grid gap-5 lg:grid-cols-5">
          <div className="lg:col-span-2">
            <p className="mb-2 text-xs font-medium text-slate-700">Newest model of each maker: {TEXT_NAME.original.toLowerCase()} vs {TEXT_NAME[focus]}</p>
            <PairChart aName={TEXT_NAME.original} bName={TEXT_NAME[focus]}
              rows={flagships.map(m => ({
                key: m.id, label: <span title={m.id}>{m.name}</span>, a: m.tokens?.original ?? null, b: m.tokens?.[focus] ?? null,
                tip: <>{m.counter}<br />{fmtUsd(m.input, 2)} / {fmtUsd(m.output, 2)} per M tokens · {TEXT_NAME[focus]} {fmtUsd(m.cost?.[focus] ?? 0)} per call</>,
              }))}
              caption="Input tokens. The same text costs a different number of tokens on each tokenizer." />
          </div>
          <div className="lg:col-span-3">
            {bothForms && (
              <p className="mb-2 rounded-md bg-emerald-50 px-2.5 py-1.5 text-xs text-emerald-900">
                Variant B form, per tokenizer: plain text is shorter on <b>{formTally.plain}</b> models, CSV on <b>{formTally.csv}</b>
                {formTally.tie > 0 && <>, a tie on {formTally.tie}</>}. Send <b>{formTally.plain >= formTally.csv ? 'plain text' : 'CSV'}</b> unless your model's row says otherwise.
              </p>
            )}
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              <select value={focus} onChange={e => setFocus(e.target.value as MarketText)} className="rounded-md border border-slate-300 px-1.5 py-1">
                {market.texts.map(t => <option key={t} value={t}>Chart & cost: {TEXT_NAME[t]}</option>)}
              </select>
              <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search models"
                className="w-44 rounded-md border border-slate-300 px-2 py-1" />
              <select value={sort} onChange={e => setSort(e.target.value as typeof sort)} className="rounded-md border border-slate-300 px-1.5 py-1">
                <option value="cost">Cheapest {TEXT_NAME[focus]} first</option><option value="tokens">Fewest tokens first</option><option value="name">By name</option>
              </select>
              <label className="flex items-center gap-1.5 text-slate-600"><input type="checkbox" checked={countedOnly} onChange={e => setCountedOnly(e.target.checked)} />counted only</label>
              <span className="ml-auto text-slate-500">{rows.length} models</span>
            </div>
            <div className="max-h-[420px] overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-white text-[10px] uppercase tracking-wide text-slate-500">
                  <tr><th className="py-1.5 pr-2">Model</th>
                    {market.texts.map(t => <th key={t} className={`pr-2 text-right ${t === focus ? 'text-indigo-700' : ''}`}>{TEXT_NAME[t]}</th>)}
                    <th className="pr-2 text-right">{TEXT_NAME[focus]} / call</th>{bothForms && <th className="pr-2">Shorter B</th>}<th>Counted by</th></tr>
                </thead>
                <tbody>
                  {(all ? rows : rows.slice(0, 80)).map(m => (
                    <tr key={m.id} className="border-t border-slate-100 align-top">
                      <td className="py-1 pr-2"><div className="font-medium text-slate-800">{m.name}</div><div className="font-mono text-[10px] text-slate-400">{m.id}</div></td>
                      {market.texts.map(t => <td key={t} className={`pr-2 text-right font-mono ${t === focus ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>{m.tokens ? fmtInt(m.tokens[t] ?? 0) : '—'}</td>)}
                      <td className="pr-2 text-right font-mono text-slate-700">{m.cost ? fmtUsd(m.cost[focus] ?? 0) : '—'}</td>
                      {bothForms && <td className="pr-2">{(f => f && <Pill className={f === 'plain' ? 'bg-slate-100 text-slate-700' : f === 'csv' ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-50 text-slate-500'}>{f}</Pill>)(shorterForm(m))}</td>}
                      <td className="text-[10px] text-slate-500">{m.counter ?? <span className="text-amber-700">{m.unavailable}</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!all && rows.length > 80 && <button onClick={() => setAll(true)} className="mt-2 text-xs font-medium text-indigo-700">Show all {rows.length}</button>}
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              Claude, Gemini, Grok and Cohere count through their official APIs: set ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY or CO_API_KEY on the backend to fill those rows. Gated Hugging Face repos need HF_TOKEN.
            </p>
          </div>
        </div>
      )}
    </Card>
  )
}

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
    { key: 'B', name: `Variant B · short (${r.s7.format === 'plain' ? 'plain text' : 'CSV'})`, input: r.s7.best_tokens, answer: r.s7.answer_tokens_est, perCall: r.s7.cost.per_call, per10k: r.s7.cost.per_10k },
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
