import type { S3Result } from '../types'
import { Bar, Card, Level, Pill } from '../ui'
import { Radar } from './charts'

const TASK_NAMES: Record<string, string> = {
  code: 'Write code', debug: 'Debug', summarise: 'Summarise', extract: 'Extract', classify: 'Classify',
  qa_docs: 'Answer from documents', reasoning: 'Reasoning / maths', creative: 'Creative', translate: 'Translate',
  plan: 'Plan / analyse', chat: 'Chat',
}

export default function S3Panel({ s3 }: { s3: S3Result }) {
  const scores = Object.entries(s3.task_type.scores).sort((a, b) => b[1] - a[1])
  const a = s3.answer_length
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <Card title="Task type" subtitle="Keyword evidence (spaCy lemmas) + embedding similarity">
        <div className="text-2xl font-semibold">{TASK_NAMES[s3.task_type.type] ?? s3.task_type.type}</div>
        <div className="text-sm text-slate-500">{(s3.task_type.confidence * 100).toFixed(0)}% confidence · {s3.items} input item{s3.items === 1 ? '' : 's'}</div>
        <div className="mt-3">
          <Radar highlight={s3.task_type.type}
            axes={Object.keys(TASK_NAMES).filter(t => t in s3.task_type.scores)
              .map(t => ({ key: t, label: TASK_NAMES[t], value: s3.task_type.scores[t] }))} />
        </div>
        <div className="mt-4 space-y-1.5">
          {scores.slice(0, 6).map(([t, v]) => (
            <div key={t} className="flex items-center gap-2 text-xs">
              <span className="w-36 truncate text-slate-600">{TASK_NAMES[t] ?? t}</span>
              <Bar value={v} className={t === s3.task_type.type ? 'bg-indigo-500' : 'bg-slate-300'} />
              <span className="w-10 text-right font-mono text-slate-500">{(v * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Difficulty and risk">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-600">Difficulty</span>
          <span className="flex items-center gap-2"><span className="font-mono text-xs text-slate-500">{s3.difficulty.score}</span><Level level={s3.difficulty.level} /></span>
        </div>
        <ul className="mt-2 space-y-1 text-xs text-slate-500">
          {Object.entries(s3.difficulty.factors).map(([k, v]) => (
            <li key={k} className="flex justify-between"><span>{k.replace(/_/g, ' ')}</span><span className="font-mono">{v}</span></li>
          ))}
        </ul>
        <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-4">
          <span className="text-sm text-slate-600">Risk</span><Level level={s3.risk.level} />
        </div>
        {s3.risk.domains.length > 0 && <div className="mt-2 flex flex-wrap gap-1">{s3.risk.domains.map(d => <Pill key={d} className="bg-rose-50 text-rose-700">{d}</Pill>)}</div>}
        <ul className="mt-2 list-disc pl-4 text-xs text-slate-500">{s3.risk.reasons.map(r => <li key={r}>{r}</li>)}</ul>
        {s3.risk.level === 'high' && <p className="mt-2 text-xs text-amber-700">Variant B will stop at the safe reduction steps (L2).</p>}
      </Card>

      <Card title="Likely answer length" subtitle={a.estimated ? 'Estimate until verify mode measures it' : 'Measured'}>
        <div className="grid grid-cols-2 gap-3">
          {[['Expected', a.expected_tokens], ['Floor', a.floor], ['Budget (1.2×)', a.budget], ['max_tokens (1.5×)', a.max_tokens]].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">{k}</div>
              <div className="text-lg font-semibold tabular-nums">{v}<span className="ml-1 text-xs font-normal text-slate-400">tok</span></div>
            </div>
          ))}
        </div>
        <ul className="mt-3 space-y-0.5 text-xs text-slate-500">{a.basis.map(b => <li key={b}>· {b}</li>)}</ul>
      </Card>

      <Card className="md:col-span-2 xl:col-span-3" title={`Missing details · ${s3.gaps.length}`}
        subtitle="Each gap becomes a highlighted, editable assumption in Variant A">
        {s3.gaps.length === 0 ? <p className="text-sm text-slate-500">Nothing missing.</p> : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {s3.gaps.map((g, i) => (
              <div key={i} className="rounded-lg border border-slate-200 p-3">
                <div className="flex items-center gap-2"><Pill className="bg-indigo-50 text-indigo-700">{g.kind}</Pill><span className="text-sm font-medium">{g.message}</span></div>
                <p className="mt-2 rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-800">[Assumption, edit if needed] {g.assumption}</p>
                {g.evidence && <p className="mt-1.5 truncate font-mono text-[11px] text-slate-400">“{g.evidence}”</p>}
                {g.segment_ids.length > 0 && <p className="mt-1 font-mono text-[10px] text-slate-400">parts {g.segment_ids.join(' ')}</p>}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
