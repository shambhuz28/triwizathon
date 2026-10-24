import { useState } from 'react'
import type { S1Result } from '../types'
import { Card, MarkedText, Pill, type Mark } from '../ui'

export default function S1Panel({ s1 }: { s1: S1Result }) {
  const [showOriginals, setShowOriginals] = useState(false)
  const marks: Mark[] = [
    ...s1.injections.map((f, i) => ({
      start: f.start, end: f.end, key: `inj${i}`,
      className: 'bg-rose-200 underline decoration-rose-500 decoration-wavy underline-offset-4',
      title: `Embedded instruction · score ${f.score} · ${f.reasons.join('; ')}`,
    })),
    ...s1.pii.map((p, i) => ({
      start: p.start, end: p.end, key: `pii${i}`, className: 'rounded bg-purple-200 px-0.5',
      title: `${p.type} · score ${p.score}${s1.pii_redacted ? ` · was "${p.text}"` : ` · would become ${p.placeholder}`}`,
    })),
  ]

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2" title="Cleaned text"
        subtitle="Every later stage works on this text. Hover a highlight for details.">
        <div className="max-h-[480px] overflow-auto rounded-lg bg-slate-50 p-4">
          <MarkedText text={s1.clean_text} marks={marks} />
        </div>
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500">
          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded bg-purple-200" />personal information</span>
          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded bg-rose-200" />embedded instruction (fenced as data)</span>
        </div>
      </Card>

      <div className="space-y-4">
        <Card title="Language">
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-semibold">{s1.language.name}</span>
            <span className="text-sm text-slate-500">{(s1.language.confidence * 100).toFixed(0)}% · {s1.language.script} script</span>
          </div>
          {!s1.language.is_english && <p className="mt-2 text-xs text-amber-700">Name detection is English-only; results may be less reliable.</p>}
          {s1.language.candidates.length > 1 && (
            <p className="mt-2 text-xs text-slate-500">Also possible: {s1.language.candidates.slice(1).map(c => `${c.code} ${(c.prob * 100).toFixed(0)}%`).join(', ')}</p>
          )}
        </Card>

        <Card title="Normalisation" subtitle="ftfy + Unicode clean-up (code blocks keep their spacing)">
          {s1.changes.length === 0 ? <p className="text-sm text-slate-500">Text was already clean.</p> : (
            <ul className="space-y-1.5 text-sm">
              {s1.changes.map(c => (
                <li key={c.kind} className="flex justify-between gap-2">
                  <span className="text-slate-700">{c.kind.replace(/_/g, ' ')}{c.detail && <span className="block text-xs text-slate-400">{c.detail}</span>}</span>
                  <span className="font-mono text-slate-500">{c.count}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title={`Personal information · ${s1.pii.length}`}
          subtitle={s1.pii_redacted ? 'Swapped for placeholders; swapped back in the final prompts' : 'Detected, not redacted (turn on redaction to swap)'}
          right={s1.pii.length > 0 && (
            <button onClick={() => setShowOriginals(v => !v)} className="text-xs font-medium text-indigo-600 hover:underline">
              {showOriginals ? 'Hide values' : 'Show values'}
            </button>
          )}>
          {s1.pii.length === 0 ? <p className="text-sm text-slate-500">None found.</p> : (
            <ul className="space-y-1.5 text-sm">
              {s1.pii.map((p, i) => (
                <li key={i} className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs text-purple-700">{p.placeholder}</span>
                  <span className="truncate text-xs text-slate-500">{showOriginals ? p.text : '••••••'}</span>
                  <Pill>{p.score.toFixed(2)}</Pill>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title={`Embedded instructions · ${s1.injections.length}`} subtitle="Patterns + similarity to known attacks">
          {s1.injections.length === 0 ? <p className="text-sm text-slate-500">None found.</p> : (
            <ul className="space-y-3">
              {s1.injections.map((f, i) => (
                <li key={i} className="rounded-lg border border-rose-200 bg-rose-50 p-3">
                  <div className="flex items-center justify-between">
                    <Pill className="bg-rose-600 text-white">score {f.score.toFixed(2)}</Pill>
                    <span className="text-xs text-rose-700">{f.in_data ? 'inside pasted data' : 'in instructions'}</span>
                  </div>
                  <p className="mt-2 font-mono text-xs text-slate-700">{f.text}</p>
                  <ul className="mt-2 list-disc pl-4 text-xs text-slate-600">{f.reasons.map(r => <li key={r}>{r}</li>)}</ul>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  )
}
