import { Fragment, useEffect, useState } from 'react'
import type { AnalyzeResponse, CompareResponse, LiveProvider, LiveResult, LiveStatus, MarketResponse } from '../types'
import { Bar, Card, Pill, fmtInt, fmtUsd } from '../ui'
import { BarChart, PairChart, SubTabs } from './charts'
import { MarketCard, RecommendationCard } from './variants'

export interface MarketState { data: MarketResponse | null; loading: boolean; error: string | null; retry: () => void }
export interface Analyzed { prompt: string; docIds: string[] }

interface LivePrompt { key: string; name: string; text: string; docIds: string[]; estimate: number }
type Cell = { state: 'running' } | { state: 'done'; result: LiveResult } | { state: 'error'; error: string }
const PROVIDERS: LiveProvider[] = ['gemini', 'ollama']
const PROVIDER_NAME: Record<LiveProvider, string> = { gemini: 'Gemini', ollama: 'Ollama' }

function livePrompts(r: AnalyzeResponse, analyzed: Analyzed): LivePrompt[] {
  const docs = r.s0?.used ?? false
  return [
    { key: 'original', name: docs ? 'Original (whole files + prompt)' : 'Original', text: analyzed.prompt,
      docIds: docs ? analyzed.docIds : [], estimate: r.s6.baseline.tokens },
    ...(docs ? [{ key: 'retrieved', name: 'After retrieval', text: r.s1.clean_text, docIds: [], estimate: r.s0!.tokens_after }] : []),
    { key: 'A', name: 'Variant A', text: r.s6.prompt, docIds: [], estimate: r.s6.tokens },
    { key: 'B_plain', name: 'Variant B · plain text', text: r.s7.prompt, docIds: [], estimate: r.s7.tokens },
    { key: 'B_csv', name: 'Variant B · CSV', text: r.s7.csv, docIds: [], estimate: r.s7.csv_tokens },
  ]
}

/** Run each prompt on Gemini and the local Ollama model and show the tokens they really used. */
function LiveRunCard({ r, analyzed, market }: { r: AnalyzeResponse; analyzed: Analyzed; market: MarketResponse | null }) {
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [cells, setCells] = useState<Record<string, Cell>>({})
  const [running, setRunning] = useState(false)
  const [compared, setCompared] = useState<Partial<Record<LiveProvider, CompareResponse>>>({})
  const [shown, setShown] = useState<LiveProvider>('ollama')
  const prompts = livePrompts(r, analyzed)

  useEffect(() => { fetch('/api/v1/live/status').then(x => x.json()).then(setStatus).catch(() => setStatus(null)) }, [])

  const providers = PROVIDERS.filter(p => status?.[p].available)
  const cell = (p: LiveProvider, k: string) => cells[`${p}:${k}`]
  const done = (p: LiveProvider, k: string) => { const c = cell(p, k); return c?.state === 'done' ? c.result : null }

  async function runAll() {
    setRunning(true)
    setCells({})
    setCompared({})
    // providers side by side, prompts one after another on each (Ollama runs locally, one at a time)
    await Promise.all(providers.map(async p => {
      let pinned: string | null = null          // Gemini: every prompt of a run goes to the same model
      const replies: Record<string, string> = {}
      for (const lp of prompts) {
        setCells(c => ({ ...c, [`${p}:${lp.key}`]: { state: 'running' } }))
        try {
          const x = await fetch('/api/v1/live/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider: p, prompt: lp.text, document_ids: lp.docIds, model: pinned }) })
          if (!x.ok) throw new Error((await x.json().catch(() => null))?.detail ?? `HTTP ${x.status}`)
          const result: LiveResult = await x.json()
          pinned = result.model
          replies[lp.key] = result.reply
          setCells(c => ({ ...c, [`${p}:${lp.key}`]: { state: 'done', result } }))
        } catch (e) {
          setCells(c => ({ ...c, [`${p}:${lp.key}`]: { state: 'error', error: e instanceof Error ? e.message : String(e) } }))
        }
      }
      if (Object.keys(replies).length) {
        const x = await fetch('/api/v1/live/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ replies, reference: 'original', labels: [...new Set(r.s5.label_sets.flatMap(l => l.items))],
            formats: [...new Set(r.s5.spans.filter(s => s.category === 'format').map(s => s.text.toLowerCase()))],
            budget: r.s3.answer_length.budget }) }).catch(() => null)
        if (x?.ok) { const cmp: CompareResponse = await x.json(); setCompared(c => ({ ...c, [p]: cmp })) }
      }
    }))
    setRunning(false)
  }

  // which Gemini model the run really used (a model whose quota ran out is skipped)
  const geminiRuns = prompts.map(lp => done('gemini', lp.key)).filter((x): x is LiveResult => x !== null)
  const geminiModels = [...new Set(geminiRuns.map(x => x.model))]
  const geminiSkipped = [...new Set(geminiRuns.flatMap(x => x.fallback_from))]
  const geminiNote = geminiSkipped.length || geminiModels.length > 1
    ? `Gemini: ${geminiSkipped.length ? `${geminiSkipped.join(', ')} had no quota left or is unavailable, so ` : ''}this run used ${geminiModels.join(', ')}.` +
      (geminiModels.length > 1 ? ' The prompts ran on different models, so compare them with care.' : '')
    : null
  // Gemini's live price from the catalogue, for the model actually used
  const geminiModel = geminiModels[0] ?? status?.gemini.model
  const geminiPrice = market?.models.find(m => geminiModel && m.id === `google/${geminiModel}`)
  const total = (x: LiveResult | null) => x ? x.input_tokens + x.output_tokens + x.thinking_tokens : null
  const verdicts = providers.flatMap(p => {
    const plain = done(p, 'B_plain'), csv = done(p, 'B_csv'), orig = done(p, 'original')
    const best = r.s7.format === 'plain' ? plain : csv
    const out: string[] = []
    if (plain && csv) {
      const d = csv.input_tokens - plain.input_tokens
      out.push(d === 0 ? `${PROVIDER_NAME[p]}: plain text and CSV both take ${plain.input_tokens} input tokens`
        : `${PROVIDER_NAME[p]}: ${d > 0 ? 'plain text' : 'CSV'} is ${Math.abs(d)} input tokens shorter (${plain.input_tokens} vs ${csv.input_tokens})` +
          ((d > 0) === (r.s7.format === 'plain') ? ', as S7 recommended' : ', unlike the o200k count S7 used'))
    }
    if (orig && best) {
      const pct = (a: number, b: number) => { const s = 1 - a / Math.max(b, 1); return `${s >= 0 ? '−' : '+'}${Math.abs(s * 100).toFixed(0)}%` }
      const vis = (x: LiveResult) => x.input_tokens + x.output_tokens
      out.push(`${PROVIDER_NAME[p]}: Variant B used ${fmtInt(vis(best))} visible tokens (input + answer) vs ${fmtInt(vis(orig))} for the original (${pct(vis(best), vis(orig))})` +
        (best.thinking_tokens || orig.thinking_tokens
          ? `; with hidden thinking ${fmtInt(total(best)!)} vs ${fmtInt(total(orig)!)} (${pct(total(best)!, total(orig)!)}), which varies from run to run`
          : ''))
    }
    return out
  })

  return (
    <Card title="Live run on Gemini and Ollama"
      subtitle="Each prompt is really sent; the tokens are what the model reports (Gemini usageMetadata, Ollama prompt_eval_count / eval_count)"
      right={<button onClick={runAll} disabled={running || providers.length === 0}
        className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">
        {running ? 'Running…' : `Run ${prompts.length} prompts`}
      </button>}>
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        {PROVIDERS.map(p => (
          <span key={p} className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 ${status?.[p].available ? 'bg-emerald-50 text-emerald-800' : 'bg-slate-100 text-slate-500'}`}>
            <span aria-hidden>{status?.[p].available ? '●' : '○'}</span>
            {PROVIDER_NAME[p]} {status?.[p].model && <span className="font-mono">{status[p].model}</span>}
            {!status?.[p].available && <span>· {status?.[p].why ?? 'checking…'}</span>}
          </span>
        ))}
      </div>
      <p className="mb-3 rounded-md bg-amber-50 px-2.5 py-1.5 text-[11px] text-amber-800">
        Running sends these prompts{r.s0?.used ? ' (including the attached files)' : ''} to Google Gemini, billed to your key, and to the local Ollama model. Ollama's input count includes its chat template; Gemini's total includes hidden thinking tokens.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-[10px] uppercase tracking-wide text-slate-500">
            <tr><th className="py-1.5 pr-2">Prompt</th><th className="pr-2 text-right">o200k est.</th>
              {providers.map(p => <th key={p} className="pr-2 text-right">{PROVIDER_NAME[p]} in · out · think · s</th>)}</tr>
          </thead>
          <tbody>
            {prompts.map(lp => (
              <Fragment key={lp.key}>
                <tr className="border-t border-slate-100">
                  <td className="py-1.5 pr-2 font-medium text-slate-800">{lp.name}</td>
                  <td className="pr-2 text-right font-mono text-slate-500">{fmtInt(lp.estimate)}</td>
                  {providers.map(p => {
                    const c = cell(p, lp.key)
                    return (
                      <td key={p} className="pr-2 text-right font-mono">
                        {!c ? <span className="text-slate-300">—</span>
                          : c.state === 'running' ? <span className="animate-pulse text-indigo-600">running…</span>
                          : c.state === 'error' ? <span className="text-rose-600" title={c.error}>failed</span>
                          : <span className="text-slate-800"><b>{fmtInt(c.result.input_tokens)}</b> · {fmtInt(c.result.output_tokens)} · {fmtInt(c.result.thinking_tokens)} · {c.result.seconds}</span>}
                      </td>
                    )
                  })}
                </tr>
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {providers.length > 0 && Object.values(cells).some(c => c.state === 'done') && (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <div>
            <p className="mb-2 text-xs font-medium text-slate-700">Measured input tokens</p>
            <PairChart aName={PROVIDER_NAME[providers[0]]} bName={providers[1] ? PROVIDER_NAME[providers[1]] : ''}
              rows={prompts.map(lp => ({ key: lp.key, label: lp.name, a: done(providers[0], lp.key)?.input_tokens ?? null,
                b: providers[1] ? done(providers[1], lp.key)?.input_tokens ?? null : null,
                tip: <>o200k estimate {fmtInt(lp.estimate)}</> }))} />
          </div>
          <div>
            <p className="mb-2 text-xs font-medium text-slate-700">Measured total tokens (input + answer + thinking)</p>
            <PairChart aName={PROVIDER_NAME[providers[0]]} bName={providers[1] ? PROVIDER_NAME[providers[1]] : ''}
              rows={prompts.map(lp => ({ key: lp.key, label: lp.name, a: total(done(providers[0], lp.key)),
                b: providers[1] ? total(done(providers[1], lp.key)) : null,
                tip: geminiPrice && done('gemini', lp.key)
                  ? <>Gemini cost {fmtUsd((done('gemini', lp.key)!.input_tokens * geminiPrice.input + (done('gemini', lp.key)!.output_tokens + done('gemini', lp.key)!.thinking_tokens) * geminiPrice.output) / 1e6)} at live prices</>
                  : undefined }))}
              caption="The answer side is where the variants differ most: Variant B caps it, Variant A fixes its format." />
          </div>
        </div>
      )}
      {verdicts.length > 0 && (
        <ul className="mt-4 space-y-1 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900">
          {verdicts.map(v => <li key={v}>· {v}</li>)}
        </ul>
      )}
      {[...new Set(Object.values(cells).flatMap(c => c.state === 'error' ? [c.error] : []))].map(e => (
        <p key={e} className="mt-2 rounded-md bg-rose-50 px-2.5 py-1.5 text-[11px] text-rose-700">{e}</p>
      ))}
      {geminiNote && <p className="mt-2 rounded-md bg-amber-50 px-2.5 py-1.5 text-[11px] text-amber-800">{geminiNote}</p>}

      {providers.some(p => prompts.some(lp => done(p, lp.key))) && (
        <OutputsSection r={r} prompts={prompts} providers={providers} shown={providers.includes(shown) ? shown : providers[0]}
          setShown={setShown} done={done} compared={compared} />
      )}
    </Card>
  )
}

/** Every prompt's reply side by side, with how it compares with the original prompt's reply. */
function OutputsSection({ r, prompts, providers, shown, setShown, done, compared }: {
  r: AnalyzeResponse; prompts: LivePrompt[]; providers: LiveProvider[]; shown: LiveProvider; setShown: (p: LiveProvider) => void
  done: (p: LiveProvider, k: string) => LiveResult | null; compared: Partial<Record<LiveProvider, CompareResponse>>
}) {
  const cmp = compared[shown]?.replies
  const ref = cmp?.original
  const labels = [...new Set(r.s5.label_sets.flatMap(l => l.items))]
  const check = (ok: boolean, label: string) => (
    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${ok ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`}>
      <span aria-hidden>{ok ? '✓' : '✗'}</span>{label}
    </span>
  )
  return (
    <section className="mt-6 border-t border-slate-200 pt-4">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h4 className="text-sm font-semibold text-slate-900">Outputs side by side</h4>
        <div className="flex rounded-md bg-slate-100 p-0.5 text-xs">
          {providers.map(p => (
            <button key={p} onClick={() => setShown(p)}
              className={`rounded px-2.5 py-1 font-medium ${shown === p ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>{PROVIDER_NAME[p]}</button>
          ))}
        </div>
        <span className="text-[11px] text-slate-500">
          Similarity: how close each reply's meaning is to the original prompt's reply (MiniLM cosine) · format and labels: what the prompt asked for
        </span>
      </div>

      {cmp && (
        <div className="mb-4 grid gap-5 lg:grid-cols-2">
          <div>
            <p className="mb-2 text-xs font-medium text-slate-700">Answer length (o200k tokens of the reply)</p>
            <BarChart labelWidth="w-44" rows={prompts.filter(lp => cmp[lp.key]).map(lp => ({
              key: lp.key, label: lp.name, value: cmp[lp.key].tokens, tone: lp.key === 'original' ? 'base' : 'step',
              note: ref && lp.key !== 'original' ? `${cmp[lp.key].tokens <= ref.tokens ? '−' : '+'}${Math.abs((1 - cmp[lp.key].tokens / Math.max(ref.tokens, 1)) * 100).toFixed(0)}%` : undefined,
              tip: `${cmp[lp.key].words} words${r.s3.answer_length.budget ? ` · budget ${r.s3.answer_length.budget}` : ''}`,
            }))} />
          </div>
          <div>
            <p className="mb-2 text-xs font-medium text-slate-700">Same answer as the original prompt? (similarity to its reply)</p>
            <BarChart labelWidth="w-44" max={1} format={v => `${(v * 100).toFixed(0)}%`} unit="cosine similarity"
              rows={prompts.filter(lp => cmp[lp.key] && lp.key !== 'original').map(lp => ({
                key: lp.key, label: lp.name, value: Math.max(0, cmp[lp.key].similarity ?? 0), tone: 'accent',
                tip: (cmp[lp.key].similarity ?? 0) >= 0.8 ? 'Says essentially the same thing' : (cmp[lp.key].similarity ?? 0) >= 0.6 ? 'Same topic, different wording or detail' : 'A noticeably different answer',
              }))} />
          </div>
        </div>
      )}

      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(prompts.length, 4)}, minmax(0, 1fr))` }}>
        {prompts.map(lp => {
          const res = done(shown, lp.key)
          const c = cmp?.[lp.key]
          return (
            <div key={lp.key} className={`flex min-w-0 flex-col rounded-lg border ${lp.key === 'original' ? 'border-slate-300' : 'border-slate-200'}`}>
              <div className="border-b border-slate-100 bg-slate-50 px-3 py-2">
                <div className="text-xs font-semibold text-slate-800">{lp.name}</div>
                {res && <div className="font-mono text-[10px] text-slate-500">{res.model} · {fmtInt(res.output_tokens)} out{res.thinking_tokens ? ` + ${fmtInt(res.thinking_tokens)} thinking` : ''} · {res.seconds}s</div>}
              </div>
              {c && (
                <div className="flex flex-wrap gap-1 border-b border-slate-100 px-3 py-1.5">
                  {c.similarity != null && lp.key !== 'original' && (
                    <span className="flex w-full items-center gap-2 text-[10px] text-slate-500">similarity <Bar value={Math.max(0, c.similarity)} className="bg-indigo-500" /><span className="font-mono">{(c.similarity * 100).toFixed(0)}%</span></span>
                  )}
                  {Object.entries(c.format_ok).map(([f, ok]) => <Fragment key={f}>{check(ok, f)}</Fragment>)}
                  {labels.length > 0 && check(c.labels_missing.length === 0, c.labels_missing.length ? `labels missing: ${c.labels_missing.join(', ')}` : 'labels')}
                  {check(!c.preamble, c.preamble ? 'chatty opening' : 'no preamble')}
                  {check(!c.over_budget, c.over_budget ? 'over budget' : 'within budget')}
                </div>
              )}
              <div className="prompt-text max-h-80 flex-1 overflow-auto px-3 py-2 text-[12px]">
                {res ? res.reply || <span className="text-slate-400">(empty reply)</span> : <span className="text-slate-400">no reply</span>}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

type Tab = 'recommend' | 'models' | 'live'

export default function ComparePanel({ r, market, analyzed }: { r: AnalyzeResponse; market: MarketState; analyzed: Analyzed | null }) {
  const [tab, setTab] = useState<Tab>('recommend')
  return (
    <div className="space-y-4">
      <SubTabs value={tab} onChange={setTab} tabs={[
        { id: 'recommend', label: 'Recommendation' },
        { id: 'models', label: 'All models', badge: market.loading ? <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" /> : market.data && <Pill>{market.data.counted}</Pill> },
        { id: 'live', label: 'Live run · Gemini + Ollama' },
      ]} />
      {tab === 'recommend' && <div className="grid gap-4 lg:grid-cols-3"><RecommendationCard r={r} /></div>}
      {tab === 'models' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <MarketCard market={market.data} loading={market.loading} error={market.error} onRetry={market.retry}
            defaultFocus={r.recommendation.choice === 'A' ? 'variant_a' : r.s7.format === 'csv' ? 'variant_b_csv' : 'variant_b'} />
        </div>
      )}
      {tab === 'live' && (analyzed ? <LiveRunCard key={`${r.total_ms}-${analyzed.prompt.length}`} r={r} analyzed={analyzed} market={market.data} />
        : <p className="text-sm text-slate-500">Analyse a prompt first.</p>)}
    </div>
  )
}
