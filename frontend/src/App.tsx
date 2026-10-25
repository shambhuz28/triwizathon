import { useEffect, useState } from 'react'
import type { AnalyzeResponse, DocumentInfo, MarketResponse, ModelInfo, RagMode } from './types'
import { SAMPLES } from './samples'
import { Stat, fmtInt, fmtUsd } from './ui'
import Documents from './Documents'
import S0Panel from './stages/S0Panel'
import S1Panel from './stages/S1Panel'
import S2Panel from './stages/S2Panel'
import S3Panel from './stages/S3Panel'
import S4Panel from './stages/S4Panel'
import S5Panel from './stages/S5Panel'
import S6Panel from './stages/S6Panel'
import S7Panel from './stages/S7Panel'
import ComparePanel from './stages/ComparePanel'

const STAGES = [
  { id: 'S0', name: 'Retrieve', blurb: 'Relevant passages from attached files (FAISS)' },
  { id: 'S1', name: 'Clean up', blurb: 'Normalise, language, personal info, injected orders' },
  { id: 'S2', name: 'Break into parts', blurb: 'Blocks, sentences, labels, repeats' },
  { id: 'S3', name: 'Understand the task', blurb: 'Type, difficulty, risk, gaps, answer length' },
  { id: 'S4', name: 'Count and profile', blurb: 'Tokens, surprisal heat map, quality, warnings' },
  { id: 'S5', name: 'Lock and compress', blurb: 'Lock key words, LLMLingua-2 removes the rest of the waste' },
  { id: 'S6', name: 'Variant A', blurb: 'Detailed: sections, exact format, constraints, self-check' },
  { id: 'S7', name: 'Variant B', blurb: 'Short: reduction ladder, answer cap, plain text or CSV' },
  { id: 'S10', name: 'Compare', blurb: 'Recommendation, every model, live Gemini + Ollama run' },
] as const
type StageId = typeof STAGES[number]['id']

export default function App() {
  const [prompt, setPrompt] = useState(SAMPLES[0].text)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('claude-sonnet-5-5')
  const [volume, setVolume] = useState(1_000_000)
  const [redact, setRedact] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<StageId>('S2')
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [ragMode, setRagMode] = useState<RagMode>('auto')
  const [market, setMarket] = useState<MarketResponse | null>(null)
  const [marketLoading, setMarketLoading] = useState(false)
  const [marketError, setMarketError] = useState<string | null>(null)
  const [analyzed, setAnalyzed] = useState<{ prompt: string; docIds: string[] } | null>(null)

  // Counts on every model take a while (live catalogue, vendor counters), so they load after the analysis.
  async function loadMarket(res: AnalyzeResponse, docIds: string[], userPrompt: string) {
    setMarket(null)
    setMarketError(null)
    setMarketLoading(true)
    try {
      const m = await fetch('/api/v1/market-tokens', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userPrompt, document_ids: res.s0?.used ? docIds : [], retrieved: res.s0?.used ? res.s1.clean_text : null,
          variant_a: res.s6.prompt, variant_b: res.s7.prompt, variant_b_csv: res.s7.csv,
          answers: { original: res.s4.cost.output_tokens_est, retrieved: res.s4.cost.output_tokens_est,
            variant_a: res.s6.answer_tokens_est, variant_b: res.s7.answer_tokens_est, variant_b_csv: res.s7.answer_tokens_est },
        }),
      })
      if (!m.ok) throw new Error(`${m.status} ${await m.text()}`)
      setMarket(await m.json())
    } catch (e) {
      setMarketError(e instanceof Error ? e.message : String(e))
    } finally {
      setMarketLoading(false)
    }
  }

  useEffect(() => {
    fetch('/api/v1/models').then(r => r.json()).then(d => { setModels(d.models); setModel(d.default) }).catch(() => {})
    fetch('/api/v1/documents').then(r => r.json()).then(setDocs).catch(() => {})
  }, [])

  async function run() {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/v1/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt, target_model: model, redact_pii: redact, monthly_volume: volume,
          document_ids: docs.map(d => d.id), rag_mode: ragMode,
        }),
      })
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
      const res: AnalyzeResponse = await r.json()
      setResult(res)
      if (!res.s0 && tab === 'S0') setTab('S2')
      setAnalyzed({ prompt, docIds: docs.map(d => d.id) })
      void loadMarket(res, docs.map(d => d.id), prompt)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const r = result
  const marketState = { data: market, loading: marketLoading, error: marketError,
    retry: () => { if (r && analyzed) void loadMarket(r, analyzed.docIds, analyzed.prompt) } }
  const stages =r?.s0 ? STAGES : STAGES.filter(s => s.id !== 'S0')
  const timing = (id: string) => r?.timings.find(t => t.stage === id)?.ms
  const warnCount = (id: StageId) => {
    if (!r) return 0
    return { S0: r.s0?.missing.length ?? 0, S1: r.s1.injections.length + (r.s1.pii_redacted ? 0 : r.s1.pii.length), S2: r.s2.duplicates.length, S3: r.s3.gaps.length, S4: r.s4.warnings.length, S5: r.s5.compression.missing_locks.length, S6: r.s6.checks.missing_locks.length + r.s6.checks.requirements_missing.length, S7: r.s7.checks.missing_locks.length + r.s7.checks.requirements_missing.length, S10: 0 }[id]
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-4 sm:px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white">OP</div>
          <div>
            <h1 className="text-lg font-semibold leading-tight text-slate-900">OptimusPrompt</h1>
            <p className="text-xs text-slate-500">Analysis pipeline · stages S0–S10 · local models; live runs only when you ask</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-5 px-4 py-6 sm:px-6">
        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <label htmlFor="prompt" className="text-sm font-semibold text-slate-900">Your prompt</label>
            <div className="flex flex-wrap gap-1.5">
              {SAMPLES.map(s => (
                <button key={s.name} onClick={() => { setPrompt(s.text); setRedact(s.name.includes('PII')) }}
                  className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-200">{s.name}</button>
              ))}
            </div>
          </div>
          <textarea id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} rows={9} spellCheck={false}
            className="w-full resize-y rounded-lg border border-slate-300 bg-slate-50 p-3 font-mono text-[13px] leading-6 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100" />
          <div className="mt-3 flex flex-wrap items-end gap-4">
            <label className="text-xs text-slate-600">Target model
              <select value={model} onChange={e => setModel(e.target.value)} className="mt-1 block rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm">
                {(models.length ? models : [{ id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' } as ModelInfo]).map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
            <label className="text-xs text-slate-600">Calls per month
              <input type="number" min={0} step={10000} value={volume} onChange={e => setVolume(Math.max(0, Number(e.target.value)))}
                className="mt-1 block w-36 rounded-md border border-slate-300 px-2 py-1.5 text-sm tabular-nums" />
            </label>
            <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
              <input type="checkbox" checked={redact} onChange={e => setRedact(e.target.checked)} className="h-4 w-4 accent-indigo-600" />
              Swap personal info for placeholders
            </label>
            <button onClick={run} disabled={loading || !prompt.trim()}
              className="ml-auto rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">
              {loading ? 'Analysing…' : 'Analyse prompt'}
            </button>
          </div>
          <Documents docs={docs} setDocs={setDocs} mode={ragMode} setMode={setRagMode} />
          {error && <p className="mt-3 rounded-md bg-rose-50 p-2 text-sm text-rose-700">Request failed: {error}. Is the backend running on :8000?</p>}
        </section>

        {r && (
          <>
            {r.s0?.used && (
              <button onClick={() => setTab('S0')}
                className="flex w-full flex-wrap items-center gap-x-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-left text-sm text-emerald-900 hover:border-emerald-300">
                <span className="font-semibold">S0 added {r.s0.passages.length} passage{r.s0.passages.length === 1 ? '' : 's'} ({fmtInt(r.s0.context_tokens)} tokens)</span>
                <span className="text-emerald-700">from {[...new Set(r.s0.passages.map(p => p.filename))].join(', ')} in front of your prompt</span>
                <span className="ml-auto text-xs font-medium text-emerald-700">See S0 →</span>
              </button>
            )}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
              {r.s0?.used
                ? <Stat label="Input tokens" value={<>{fmtInt(r.s0.tokens_before)} <span className="text-slate-400">→</span> {fmtInt(r.s0.tokens_after)}</>}
                    sub={`whole files + prompt → after retrieval (−${((1 - r.s0.tokens_after / Math.max(r.s0.tokens_before, 1)) * 100).toFixed(0)}%)`} tone="good" />
                : <Stat label="Input tokens" value={fmtInt(r.s4.primary.count)} sub={r.s4.primary.exact ? 'exact' : `${r.s4.primary.tokenizer} ±20%`} />}
              <Stat label="Padding + repeats" value={fmtInt(r.s4.removable_tokens)} sub={`${((r.s4.removable_tokens / Math.max(r.s4.primary.count, 1)) * 100).toFixed(0)}% of the prompt`} tone={r.s4.removable_tokens ? 'bad' : 'good'} />
              <Stat label="Quality" value={`${r.s4.quality.overall}/100`} sub={`${r.s4.warnings.length} warnings`} tone={r.s4.quality.overall >= 75 ? 'good' : 'default'} />
              <Stat label="Task" value={r.s3.task_type.type} sub={`${r.s3.difficulty.level} difficulty · ${r.s3.risk.level} risk`} />
              <Stat label="After S5" value={fmtInt(r.s5.compression.compressed_tokens)}
                sub={`−${((1 - r.s5.compression.ratio) * 100).toFixed(0)}% · ${r.s5.lock_set.length} locks ${r.s5.compression.locks_preserved ? 'kept' : 'LOST'}`}
                tone={r.s5.compression.locks_preserved ? 'good' : 'bad'} />
              <Stat label="Cost / month" value={fmtUsd(r.s4.cost.per_month, 0)} sub={`${fmtUsd(r.s4.cost.per_call)} per call`} />
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {([['S6', 'A', 'Variant A · detailed / precise', r.s6.tokens, r.s6.cost.per_call, r.s6.checks.locks_preserved],
                 ['S7', 'B', `Variant B · short (${r.s7.format === 'plain' ? 'plain text' : 'CSV'})`, r.s7.best_tokens, r.s7.cost.per_call, r.s7.checks.locks_preserved]] as const).map(([id, key, name, tokens, perCall, ok]) => (
                <button key={id} onClick={() => setTab(id)}
                  className={`flex flex-wrap items-center gap-x-3 rounded-lg border px-4 py-2 text-left text-sm text-indigo-900 hover:border-indigo-400 ${r.recommendation.choice === key ? 'border-indigo-500 bg-indigo-100 ring-1 ring-indigo-500' : 'border-indigo-200 bg-indigo-50'}`}>
                  <span className="font-semibold">{name}</span>
                  {r.recommendation.choice === key && <span className="rounded-full bg-indigo-600 px-2 py-0.5 text-[10px] font-semibold text-white">recommended</span>}
                  {r.recommendation.fallback === key && <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-indigo-700">fallback</span>}
                  <span className="font-mono text-indigo-700">{fmtInt(tokens)} tok vs {fmtInt(r.s6.baseline.tokens)} · {fmtUsd(perCall)}/call</span>
                  <span className={`text-xs ${ok ? 'text-emerald-700' : 'text-rose-700'}`}>{ok ? 'locks kept' : 'locks lost'}</span>
                  <span className="ml-auto text-xs font-medium text-indigo-700">See {id} →</span>
                </button>
              ))}
            </div>
            {r.recommendation.choice === 'original' && (
              <p className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700">
                <b>Recommended: send the original prompt.</b> {r.recommendation.reason}
              </p>
            )}
            <button onClick={() => setTab('S10')} className="-mt-1 text-left text-xs text-slate-500 hover:text-indigo-700">
              {r.recommendation.reason} · <span className="font-medium text-indigo-700">trade-offs, every model and a live Gemini + Ollama run →</span>
            </button>

            <nav className={`grid grid-cols-2 gap-2 ${stages.length > 8 ? 'sm:grid-cols-5 xl:grid-cols-9' : 'sm:grid-cols-4 xl:grid-cols-8'}`}>
              {stages.map(s => (
                <button key={s.id} onClick={() => setTab(s.id)}
                  className={`relative rounded-xl border px-3 py-2.5 text-left transition ${tab === s.id ? 'border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500' : 'border-slate-200 bg-white hover:border-slate-300'}`}>
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-bold ${tab === s.id ? 'text-indigo-700' : 'text-slate-400'}`}>{s.id}</span>
                    <span className="font-mono text-[10px] text-slate-400">{timing(s.id)?.toFixed(0)} ms</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-900">{s.name}</div>
                  <div className="text-[11px] leading-snug text-slate-500">{s.blurb}</div>
                  {warnCount(s.id) > 0 && <span className="absolute -right-1.5 -top-1.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-amber-500 px-1 text-[10px] font-bold text-white">{warnCount(s.id)}</span>}
                </button>
              ))}
            </nav>
            <p className="-mt-2 text-right text-[11px] text-slate-400">Each stage builds on the ones before · total {r.total_ms.toFixed(0)} ms</p>

            {tab === 'S0' && r.s0 && <S0Panel s0={r.s0} />}
            {tab === 'S1' && <S1Panel s1={r.s1} />}
            {tab === 'S2' && <S2Panel s2={r.s2} text={r.s1.clean_text} segmentTokens={r.s4.segment_tokens} />}
            {tab === 'S3' && <S3Panel s3={r.s3} />}
            {tab === 'S4' && <S4Panel s4={r.s4} text={r.s1.clean_text} />}
            {tab === 'S5' && <S5Panel s5={r.s5} text={r.s1.clean_text} />}
            {tab === 'S6' && <S6Panel r={r} />}
            {tab === 'S7' && <S7Panel r={r} />}
            {tab === 'S10' && <ComparePanel r={r} market={marketState} analyzed={analyzed} />}
          </>
        )}

        {!r && !loading && (
          <div className="rounded-xl border border-dashed border-slate-300 p-10 text-center text-sm text-slate-500">
            Paste a prompt (or load a sample) and press <b>Analyse prompt</b>. The first run takes a few seconds while local models load.
          </div>
        )}
      </main>
    </div>
  )
}
