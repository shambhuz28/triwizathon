import { useState } from 'react'
import type { Label, S2Result, Segment } from '../types'
import { Card, LABEL_STYLE, Pill } from '../ui'

export default function S2Panel({ s2, text, segmentTokens }: { s2: S2Result; text: string; segmentTokens: Record<string, number> }) {
  const [selected, setSelected] = useState<Segment | null>(null)
  const byId = Object.fromEntries(s2.segments.map(s => [s.id, s]))

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="The prompt, split and labelled"
        subtitle="Every sentence gets one label. Struck-through parts (padding and repeats) are marked for removal. Click a part to see why."
        right={<Pill className={s2.reconstructs ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}>
          {s2.reconstructs ? '✓ parts rebuild the input exactly' : '✗ coverage error'}
        </Pill>}>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {(Object.keys(LABEL_STYLE) as Label[]).filter(l => s2.label_counts[l]).map(l => (
            <span key={l} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${LABEL_STYLE[l].bg}`}>
              <span className={`h-2 w-2 rounded-full ${LABEL_STYLE[l].chip}`} />{LABEL_STYLE[l].name} · {s2.label_counts[l]}
            </span>
          ))}
        </div>
        <div className="prompt-text max-h-[560px] overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere] rounded-lg bg-slate-50 p-4">
          {s2.segments.map(seg => {
            const lead = text.slice(seg.start, seg.content_start)
            const trail = text.slice(seg.content_end, seg.end)
            const st = LABEL_STYLE[seg.label]
            const active = selected?.id === seg.id
            return (
              <span key={seg.id}>
                {lead}
                <span onClick={() => setSelected(seg)} title={`${seg.id} · ${st.name} · ${(seg.confidence * 100).toFixed(0)}%`}
                  className={`cursor-pointer rounded px-0.5 box-decoration-clone ${st.bg} ${seg.remove ? 'line-through decoration-slate-500/70 opacity-60' : ''} ${active ? 'ring-2 ring-indigo-500' : 'hover:ring-1 hover:ring-slate-400'}`}>
                  <sup className={`mr-1 rounded px-1 font-sans text-[9px] font-semibold text-white ${st.chip}`}>
                    {seg.requirement_id ?? seg.id}
                  </sup>
                  {seg.text}
                </span>
                {trail}
              </span>
            )
          })}
        </div>
      </Card>

      <div className="space-y-4">
        <Card title={selected ? `${selected.id} · ${LABEL_STYLE[selected.label].name}` : 'Part details'}
          subtitle={selected ? `${selected.kind}${selected.section ? ` · section “${selected.section}”` : ''} · ${segmentTokens[selected.id] ?? 0} tokens` : 'Click a part of the prompt'}>
          {selected ? (
            <div className="space-y-3 text-sm">
              <p className="rounded bg-slate-50 p-2 font-mono text-xs">{selected.text}</p>
              {selected.duplicate_of && <p className="text-xs text-amber-700">Repeats {selected.duplicate_of} (“{byId[selected.duplicate_of]?.text.slice(0, 60)}…”), which is kept.</p>}
              <div>
                <div className="mb-1 text-xs font-medium text-slate-500">Why this label</div>
                <ul className="list-disc space-y-0.5 pl-4 text-xs text-slate-600">{selected.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
              </div>
              <div>
                <div className="mb-1 text-xs font-medium text-slate-500">Label scores (speech acts + classifier)</div>
                {Object.entries(selected.scores).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([l, v]) => (
                  <div key={l} className="flex items-center gap-2 text-xs">
                    <span className="w-16 text-slate-600">{l}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                      <div className={`h-full ${LABEL_STYLE[l as Label]?.chip ?? 'bg-slate-400'}`} style={{ width: `${Math.min(100, (v / Math.max(...Object.values(selected.scores), 0.01)) * 100)}%` }} />
                    </div>
                    <span className="w-10 text-right font-mono text-slate-500">{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <p className="text-sm text-slate-500">Code, JSON, tables and tagged blocks are kept whole; the rest is split into sentences and list items, and conversational filler (“oh and please”, “lol”, “ok?”) is carved off their edges. Each part is labelled from weighted rule hits (regex cues, spaCy imperative detection, header sections) plus a vote from MiniLM similarity to labelled example sentences.</p>}
        </Card>

        <Card title={`Requirements · ${s2.requirements.length}`} subtitle="Every instruction gets an ID so it can be traced into Variant A and B">
          <ul className="space-y-1.5 text-xs">
            {s2.requirements.map(r => (
              <li key={r.id} className="flex gap-2">
                <span className={`h-fit rounded px-1.5 py-0.5 font-semibold text-white ${LABEL_STYLE[r.label].chip}`}>{r.id}</span>
                <span className="text-slate-700">{r.text}</span>
              </li>
            ))}
          </ul>
        </Card>

        {(() => {
          const pads = s2.segments.filter(s => s.label === 'padding')
          const toks = pads.reduce((n, s) => n + (segmentTokens[s.id] ?? 0), 0)
          return (
            <Card title={`Padding · ${pads.length}`} subtitle={`${toks} tokens of small talk — whole sentences, plus filler carved off the edges of instructions`}>
              {pads.length === 0 ? <p className="text-sm text-slate-500">No padding found.</p> : (
                <div className="flex flex-wrap gap-1.5">
                  {pads.map(p => (
                    <button key={p.id} onClick={() => setSelected(p)} title={p.reasons[0]}
                      className={`rounded-md px-2 py-1 font-mono text-[11px] line-through decoration-slate-400 ${p.kind === 'filler' ? 'bg-slate-100 text-slate-500' : 'bg-slate-200 text-slate-700'}`}>
                      {p.text}
                    </button>
                  ))}
                </div>
              )}
              {pads.some(p => p.kind === 'filler') && (
                <p className="mt-2 text-[11px] text-slate-400">Light chips are filler carved from a clause (“oh and please”, “lol”, “i guess”); dark chips are whole small-talk sentences.</p>
              )}
            </Card>
          )
        })()}

        <Card title={`Repeats · ${s2.duplicates.length}`} subtitle="Embedding cosine + lemma overlap; the most specific is kept">
          {s2.duplicates.length === 0 ? <p className="text-sm text-slate-500">No repeated instructions.</p> : s2.duplicates.map((d, i) => (
            <div key={i} className="mb-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs">
              <div className="font-medium text-amber-800">Keep {d.keep} · drop {d.drop.join(', ')}</div>
              <div className="mt-1 text-amber-700">{d.method}</div>
            </div>
          ))}
        </Card>

        <Card title={`Structured blocks · ${s2.blocks.length}`} subtitle="Code, JSON, tables and tag-fenced text are never split">
          {s2.blocks.length === 0 ? <p className="text-sm text-slate-500">None.</p> : (
            <ul className="space-y-1.5 text-xs">
              {s2.blocks.map(b => (
                <li key={b.id} className="flex items-center justify-between">
                  <span><Pill>{b.kind}</Pill> <span className="ml-1 font-mono text-slate-500">{JSON.stringify(b.meta).slice(0, 50)}</span></span>
                  <span className="text-slate-400">{b.end - b.start} chars</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  )
}
