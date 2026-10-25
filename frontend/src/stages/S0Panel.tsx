import type { S0Result } from '../types'
import { Bar, Card, KIND_STYLE, Pill, fmtInt } from '../ui'

export default function S0Panel({ s0 }: { s0: S0Result }) {
  const kindOf = (id: string) => s0.documents.find(d => d.id === id)?.kind ?? ''
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2"
        title={s0.used ? `Passed to S1 · ${s0.passages.length} passage${s0.passages.length === 1 ? '' : 's'} · ${fmtInt(s0.context_tokens)} tokens` : 'Nothing passed to S1'}
        subtitle="Placed in front of the prompt as one <documents> block: S1 checks it for personal info and planted instructions, S2 keeps it whole as data, S5 locks it">
        {s0.used ? (
          <ol className="space-y-3">
            {s0.passages.map((p, i) => (
              <li key={`${p.doc_id}-${p.chunks[0]}`} className="rounded-lg border border-slate-200">
                <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 bg-slate-50 px-3 py-1.5 text-xs">
                  <span className="font-mono font-semibold text-slate-500">[{i + 1}]</span>
                  <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${KIND_STYLE[kindOf(p.doc_id)] ?? 'bg-slate-200 text-slate-700'}`}>{kindOf(p.doc_id)}</span>
                  <span className="font-medium text-slate-800">{p.filename}</span>
                  {p.location && <span className="text-slate-500">· {p.location}</span>}
                  <span className="ml-auto font-mono text-slate-500">score {p.score.toFixed(2)} · {p.tokens} tok · chunk{p.chunks.length > 1 ? 's' : ''} {p.chunks.join(', ')}</span>
                </div>
                <div className="prompt-text max-h-64 overflow-auto px-3 py-2 text-[13px]">{p.text}</div>
              </li>
            ))}
          </ol>
        ) : <p className="text-sm text-slate-600">{s0.reason}</p>}
      </Card>

      <div className="space-y-4">
        <Card title="Decision" right={<Pill className={s0.used ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}>{s0.used ? 'documents used' : 'prompt unchanged'}</Pill>}>
          <p className="text-sm text-slate-700">{s0.reason}</p>
          <div className="mt-3 space-y-1.5 text-xs">
            <div className="flex items-center gap-2">
              <span className="w-24 text-slate-500">Best match</span>
              <div className="relative flex-1">
                <Bar value={Math.max(0, s0.best_score)} className={s0.best_score >= s0.threshold ? 'bg-emerald-500' : 'bg-slate-400'} />
                <span className="absolute -top-0.5 h-3 w-0.5 bg-slate-700" style={{ left: `${s0.threshold * 100}%` }} title={`needs ${s0.threshold}`} />
              </div>
              <span className="w-10 text-right font-mono text-slate-600">{s0.best_score.toFixed(2)}</span>
            </div>
            <div className="flex justify-between"><span className="text-slate-500">Needs</span><span className="font-mono">{s0.threshold.toFixed(2)} cosine, or a reference to a file</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Points at a file</span><span>{s0.explicit_reference ? `“${s0.explicit_reference}”` : 'no'}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Mode</span><span>{s0.mode}</span></div>
          </div>
          {s0.missing.length > 0 && <p className="mt-3 rounded-md bg-amber-50 p-2 text-xs text-amber-800">{s0.missing.length} attached file(s) are no longer in the store.</p>}
        </Card>

        <Card title="Tokens before and after retrieval" subtitle="o200k · without retrieval the whole files would go in with the prompt">
          <dl className="space-y-1.5 text-xs">
            {[['Prompt', s0.prompt_tokens], ['Whole files', s0.document_tokens], ['Retrieved passages', s0.context_tokens]].map(([k, v]) => (
              <div key={k} className="flex justify-between"><dt className="text-slate-500">{k}</dt><dd className="font-mono">{fmtInt(v as number)}</dd></div>
            ))}
            <div className="flex justify-between border-t border-slate-100 pt-1.5"><dt className="font-medium text-slate-700">Before: whole files + prompt</dt><dd className="font-mono font-semibold">{fmtInt(s0.tokens_before)}</dd></div>
            <div className="flex justify-between"><dt className="font-medium text-slate-700">After: passages + prompt</dt><dd className="font-mono font-semibold">{fmtInt(s0.tokens_after)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Saved by retrieval</dt>
              <dd className={`font-mono font-semibold ${s0.tokens_after < s0.tokens_before ? 'text-emerald-600' : 'text-slate-600'}`}>
                {fmtInt(s0.tokens_before - s0.tokens_after)} ({((1 - s0.tokens_after / Math.max(s0.tokens_before, 1)) * 100).toFixed(0)}%)
              </dd></div>
          </dl>
        </Card>

        <Card title={`Files searched · ${s0.documents.length}`} subtitle={`${fmtInt(s0.searched_chunks)} chunks · ${s0.model.split('/').pop()} · ${s0.index}`}>
          <ul className="space-y-1 text-xs">
            {s0.documents.map(d => (
              <li key={d.id} className="flex justify-between gap-2">
                <span className="truncate text-slate-700" title={d.filename}>{d.filename}</span>
                <span className="shrink-0 font-mono text-slate-400">{d.chunks} ch · {fmtInt(d.tokens)} tok</span>
              </li>
            ))}
          </ul>
        </Card>

        {s0.queries.length > 0 && (
          <Card title={`Queries · ${s0.queries.length}`} subtitle="The whole prompt and each of its sentences; a chunk keeps its best score">
            <ul className="max-h-40 space-y-1 overflow-auto text-xs text-slate-600">
              {s0.queries.map((q, i) => <li key={i} className="truncate" title={q}>{i === 0 ? <b>whole prompt</b> : q}</li>)}
            </ul>
          </Card>
        )}
      </div>

      {s0.candidates.length > 0 && (
        <Card className="lg:col-span-3" title="Nearest chunks" subtitle="FAISS inner product on unit-normalised MiniLM embeddings (= cosine)">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr><th className="w-40 py-1.5 pr-3 font-medium">Score</th><th className="py-1.5 pr-3 font-medium">Where</th><th className="py-1.5 pr-3 font-medium">Text</th><th className="py-1.5 font-medium" /></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {s0.candidates.map(c => (
                  <tr key={`${c.doc_id}-${c.chunk}`} className={c.selected ? '' : 'text-slate-400'}>
                    <td className="py-1.5 pr-3">
                      <div className="flex items-center gap-2"><Bar value={Math.max(0, c.score)} className={c.selected ? 'bg-indigo-500' : 'bg-slate-300'} /><span className="w-9 text-right font-mono">{c.score.toFixed(2)}</span></div>
                    </td>
                    <td className="whitespace-nowrap py-1.5 pr-3">{c.filename}{c.location && <span className="text-slate-400"> · {c.location}</span>}</td>
                    <td className="py-1.5 pr-3">{c.preview}</td>
                    <td className="py-1.5">{c.selected && <Pill className="bg-indigo-100 text-indigo-700">used</Pill>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
