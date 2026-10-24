import { useMemo, useState } from 'react'
import type { Label, S4Result, WordInfo } from '../types'
import { Bar, Card, LABEL_STYLE, MarkedText, Pill, fmtInt, fmtUsd } from '../ui'

type Signal = 'score' | 'surprisal' | 'excess' | 'entropy' | 'rarity'
const SIGNALS: { id: Signal; name: string; help: string }[] = [
  { id: 'score', name: 'Informativeness', help: 'All signals combined (z-scored mix of surprisal, excess surprise, rarity, keyphrase salience, minus echo)' },
  { id: 'surprisal', name: 'Surprisal', help: '−log₂ p(word | left context) under GPT-2 small, in bits' },
  { id: 'excess', name: 'Excess surprise', help: 'Surprisal minus the LM’s predictive entropy: surprise beyond what the model expected' },
  { id: 'entropy', name: 'Entropy', help: 'How uncertain the LM was before the word, in bits (high = open slot)' },
  { id: 'rarity', name: 'Rarity', help: '−log₂ of the word’s corpus frequency (wordfreq), independent of context' },
]

function heatColor(t: number) {
  // 0 → cool slate-blue, 1 → hot orange-red
  const h = 220 - 205 * t
  const s = 30 + 60 * t
  const l = 95 - 30 * t
  return `hsl(${h} ${s}% ${l}%)`
}

const SEV: Record<string, string> = {
  error: 'border-rose-200 bg-rose-50 text-rose-800', warn: 'border-amber-200 bg-amber-50 text-amber-800', info: 'border-sky-200 bg-sky-50 text-sky-800',
}

export default function S4Panel({ s4, text }: { s4: S4Result; text: string }) {
  const [signal, setSignal] = useState<Signal>('score')
  const [hover, setHover] = useState<WordInfo | null>(null)
  const [showLow, setShowLow] = useState(false)

  const norm = useMemo(() => {
    const vals = s4.heatmap.map(w => w[signal])
    const lo = Math.min(...vals, 0), hi = Math.max(...vals, 1e-6)
    return (v: number) => (v - lo) / (hi - lo || 1)
  }, [s4.heatmap, signal])

  const heatNodes = useMemo(() => {
    const out: React.ReactNode[] = []
    let pos = 0
    s4.heatmap.forEach((w, i) => {
      if (w.start > pos) out.push(text.slice(pos, w.start))
      const t = signal === 'score' ? w.score : norm(w[signal])
      out.push(
        <span key={i} onMouseEnter={() => setHover(w)} style={{ background: heatColor(t) }}
          className={`rounded-[3px] px-[1px] ${w.echo ? 'underline decoration-dotted decoration-slate-500' : ''} ${w.filler ? 'line-through decoration-slate-400/60' : ''}`}>
          {text.slice(w.start, w.end)}
        </span>,
      )
      pos = w.end
    })
    if (pos < text.length) out.push(text.slice(pos))
    return out
  }, [s4.heatmap, text, signal, norm])

  const q = s4.quality
  const c = s4.cost
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="Information heat map"
        subtitle={SIGNALS.find(s => s.id === signal)?.help}
        right={
          <div className="flex flex-wrap justify-end gap-1">
            {SIGNALS.map(s => (
              <button key={s.id} onClick={() => setSignal(s.id)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium ${signal === s.id ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
                {s.name}
              </button>
            ))}
          </div>
        }>
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          low<div className="h-2 w-40 rounded-full" style={{ background: `linear-gradient(90deg, ${heatColor(0)}, ${heatColor(0.5)}, ${heatColor(1)})` }} />high
          <span className="ml-3">struck = filler word · dotted = echo (repeat)</span>
          <label className="ml-auto flex items-center gap-1.5"><input type="checkbox" checked={showLow} onChange={e => setShowLow(e.target.checked)} />outline removal candidates</label>
        </div>
        <div className="mt-3 max-h-[440px] overflow-auto rounded-lg bg-slate-50 p-4" onMouseLeave={() => setHover(null)}>
          {showLow
            ? <MarkedText text={text} marks={s4.low_info.map((p, i) => ({ start: p.start, end: p.end, key: `l${i}`, className: 'rounded bg-rose-100 outline outline-1 outline-rose-400', title: `${p.reason} · score ${p.score}` }))} />
            : <div className="prompt-text">{heatNodes}</div>}
        </div>
        <div className="mt-3 min-h-[44px] rounded-lg border border-slate-200 px-3 py-2 text-xs">
          {hover ? (
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span className="font-mono font-semibold text-slate-900">“{hover.text}”</span>
              <span>informativeness <b>{hover.score.toFixed(2)}</b></span>
              <span>surprisal <b>{hover.surprisal.toFixed(1)}</b> bits</span>
              <span>entropy <b>{hover.entropy.toFixed(1)}</b></span>
              <span>excess <b>{hover.excess.toFixed(1)}</b></span>
              <span>rarity <b>{hover.rarity.toFixed(1)}</b></span>
              {hover.keyphrase > 0 && <span>keyphrase <b>{hover.keyphrase.toFixed(2)}</b></span>}
              {hover.echo && <Pill className="bg-slate-200">echo</Pill>}
              {hover.filler && <Pill className="bg-slate-200">filler</Pill>}
            </div>
          ) : <span className="text-slate-400">Hover a word to see its signals. Language model: {s4.lm}</span>}
        </div>
      </Card>

      <div className="space-y-4">
        <Card title="Tokens" subtitle={s4.primary.exact ? 'Exact count from the provider' : `Approximate (±${s4.primary.margin * 100}%) — no provider counter available`}>
          <div className="text-3xl font-semibold tabular-nums">{fmtInt(s4.primary.count)}</div>
          <ul className="mt-2 space-y-0.5 text-xs text-slate-500">
            {s4.tokens.map(t => <li key={t.tokenizer} className="flex justify-between"><span className="font-mono">{t.tokenizer}</span><span>{fmtInt(t.count)}{t.exact ? ' · exact' : ''}</span></li>)}
          </ul>
          <div className="mt-4 text-xs font-medium text-slate-500">Where the tokens go</div>
          <div className="mt-1.5 flex h-3 overflow-hidden rounded-full">
            {s4.by_label.map(b => <div key={b.label} title={`${b.label}: ${b.tokens}`} className={LABEL_STYLE[b.label as Label]?.chip} style={{ width: `${b.share * 100}%` }} />)}
          </div>
          <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
            {s4.by_label.map(b => (
              <li key={b.label} className="flex items-center justify-between">
                <span className="flex items-center gap-1.5"><span className={`h-2 w-2 rounded-full ${LABEL_STYLE[b.label as Label]?.chip}`} />{b.label}</span>
                <span className="font-mono text-slate-500">{b.tokens}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-600"><b>{s4.removable_tokens}</b> tokens are padding or repeats.</p>
        </Card>

        <Card title="Cost preview" subtitle={`${c.model_label} · prices checked ${c.last_checked}`}>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div><div className="text-[11px] text-slate-500">per call</div><div className="font-semibold tabular-nums">{fmtUsd(c.per_call)}</div></div>
            <div><div className="text-[11px] text-slate-500">per 10k</div><div className="font-semibold tabular-nums">{fmtUsd(c.per_10k, 2)}</div></div>
            <div><div className="text-[11px] text-slate-500">per month</div><div className="font-semibold tabular-nums">{fmtUsd(c.per_month, 0)}</div></div>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            {c.input_tokens} in × ${c.price_input}/M + ~{c.output_tokens_est} out × ${c.price_output}/M. The answer is
            <b> {(c.output_share_of_cost * 100).toFixed(0)}%</b> of the bill; writing it takes ~{c.answer_seconds}s. Volume {fmtInt(c.monthly_volume)}/month.
          </p>
        </Card>
      </div>

      <Card title="Most surprising words" subtitle="Highest informativeness — these carry the prompt">
        <ul className="space-y-1.5">
          {s4.surprising.map(w => (
            <li key={w.start} className="flex items-center gap-2 text-xs">
              <span className="w-28 truncate font-mono font-medium text-slate-900">{w.text}</span>
              <Bar value={w.score} className="bg-orange-500" />
              <span className="w-9 text-right font-mono text-slate-500">{w.score.toFixed(2)}</span>
            </li>
          ))}
        </ul>
        <div className="mt-3 flex flex-wrap gap-1">
          {Array.from(new Set(s4.surprising.flatMap(w => w.drivers.filter(d => !d.startsWith('in '))))).map(d => <Pill key={d}>{d}</Pill>)}
        </div>
      </Card>

      <Card title={`Low-information phrases · ${s4.low_info.length}`} subtitle="First in line for removal in Variant B (data and code are never candidates)">
        <ul className="max-h-72 space-y-1.5 overflow-auto text-xs">
          {s4.low_info.map((p, i) => (
            <li key={i} className="flex items-start justify-between gap-2">
              <span className="font-mono text-slate-700">“{p.text}”</span>
              <span className="shrink-0 text-right text-[11px] text-slate-400">{p.reason}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card title={`Prompt quality · ${q.overall}/100`}>
        <div className="space-y-2">
          {(['clarity', 'specificity', 'rules', 'structure', 'repetition'] as const).map(k => (
            <div key={k} className="flex items-center gap-2 text-xs">
              <span className="w-20 capitalize text-slate-600">{k}</span>
              <Bar value={q[k]} max={100} className={q[k] >= 75 ? 'bg-emerald-500' : q[k] >= 50 ? 'bg-amber-500' : 'bg-rose-500'} />
              <span className="w-8 text-right font-mono text-slate-500">{q[k]}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 text-xs font-medium text-slate-500">Keyphrases (YAKE + KeyBERT)</div>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {s4.keyphrases.filter(k => k.source === 'keybert').slice(0, 10).map(k => <Pill key={k.text} className="bg-indigo-50 text-indigo-700">{k.text}</Pill>)}
        </div>
      </Card>

      <Card className="lg:col-span-3" title={`Warnings · ${s4.warnings.length}`}>
        {s4.warnings.length === 0 ? <p className="text-sm text-slate-500">No problems found.</p> : (
          <div className="grid gap-2 md:grid-cols-2">
            {s4.warnings.map(w => (
              <div key={w.code} className={`rounded-lg border p-3 text-sm ${SEV[w.severity]}`}>
                <div className="flex items-center gap-2 font-medium"><span className="text-[10px] uppercase tracking-wide opacity-70">{w.severity}</span>{w.message}</div>
                <p className="mt-1 text-xs opacity-80">{w.hint}</p>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
