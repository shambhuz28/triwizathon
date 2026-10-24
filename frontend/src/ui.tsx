import type { ReactNode } from 'react'
import type { Label, LockCategory } from './types'

export const LABEL_STYLE: Record<Label, { bg: string; chip: string; name: string }> = {
  role: { bg: 'bg-violet-100', chip: 'bg-violet-600', name: 'Role' },
  task: { bg: 'bg-blue-100', chip: 'bg-blue-600', name: 'Task' },
  context: { bg: 'bg-cyan-100', chip: 'bg-cyan-600', name: 'Context' },
  data: { bg: 'bg-emerald-100', chip: 'bg-emerald-600', name: 'Data' },
  examples: { bg: 'bg-teal-100', chip: 'bg-teal-600', name: 'Examples' },
  rules: { bg: 'bg-amber-100', chip: 'bg-amber-600', name: 'Rules' },
  format: { bg: 'bg-fuchsia-100', chip: 'bg-fuchsia-600', name: 'Format' },
  question: { bg: 'bg-sky-100', chip: 'bg-sky-600', name: 'Question' },
  padding: { bg: 'bg-slate-200', chip: 'bg-slate-500', name: 'Padding' },
}

export const LOCK_STYLE: Record<LockCategory, { bg: string; dot: string; name: string }> = {
  number: { bg: 'bg-orange-200', dot: 'bg-orange-500', name: 'Numbers' },
  name: { bg: 'bg-pink-200', dot: 'bg-pink-500', name: 'Names' },
  identifier: { bg: 'bg-indigo-200', dot: 'bg-indigo-500', name: 'Identifiers' },
  restriction: { bg: 'bg-red-200', dot: 'bg-red-500', name: 'Restrictions' },
  label: { bg: 'bg-yellow-200', dot: 'bg-yellow-500', name: 'Labels' },
  code: { bg: 'bg-slate-300', dot: 'bg-slate-600', name: 'Code' },
  quote: { bg: 'bg-lime-200', dot: 'bg-lime-600', name: 'Quotes' },
  placeholder: { bg: 'bg-purple-200', dot: 'bg-purple-500', name: 'Placeholders' },
  format: { bg: 'bg-fuchsia-200', dot: 'bg-fuchsia-500', name: 'Format' },
  data: { bg: 'bg-emerald-100', dot: 'bg-emerald-500', name: 'Data' },
}

export function Card({ title, subtitle, right, children, className = '' }: {
  title?: ReactNode; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-900">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, sub, tone = 'default' }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'default' | 'good' | 'bad' }) {
  const color = tone === 'good' ? 'text-emerald-600' : tone === 'bad' ? 'text-rose-600' : 'text-slate-900'
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export function Pill({ children, className = 'bg-slate-100 text-slate-700' }: { children: ReactNode; className?: string }) {
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${className}`}>{children}</span>
}

export function Level({ level }: { level: string }) {
  const c = level === 'high' ? 'bg-rose-100 text-rose-700' : level === 'medium' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'
  return <Pill className={c}>{level}</Pill>
}

export function Bar({ value, max = 1, className = 'bg-indigo-500' }: { value: number; max?: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, (value / (max || 1)) * 100))
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
      <div className={`h-full rounded-full ${className}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export interface Mark { start: number; end: number; className: string; title?: string; onClick?: () => void; key?: string }

/** Render text with non-overlapping highlighted ranges (earlier/longer marks win on overlap). */
export function MarkedText({ text, marks, className = 'prompt-text' }: { text: string; marks: Mark[]; className?: string }) {
  const sorted = [...marks].sort((a, b) => a.start - b.start || b.end - a.end)
  const out: ReactNode[] = []
  let pos = 0
  sorted.forEach((m, i) => {
    if (m.start < pos || m.end <= m.start) return
    if (m.start > pos) out.push(text.slice(pos, m.start))
    out.push(
      <span key={m.key ?? i} title={m.title} onClick={m.onClick}
        className={`${m.className} ${m.onClick ? 'cursor-pointer' : ''}`}>
        {text.slice(m.start, m.end)}
      </span>,
    )
    pos = m.end
  })
  if (pos < text.length) out.push(text.slice(pos))
  return <div className={className}>{out}</div>
}

export const fmtUsd = (v: number, digits = 6) => `$${v.toFixed(digits)}`
export const fmtInt = (v: number) => v.toLocaleString('en-US')
