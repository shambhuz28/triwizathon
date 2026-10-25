import { useState } from 'react'
import type { ReactNode } from 'react'
import { fmtInt } from '../ui'

// Bar fills: one hue (indigo) for the prompt versions, slate for the baseline being compared
// against, a darker indigo step for the bar the panel is about. Text never takes these colours.
const FILL = { base: 'bg-slate-400', step: 'bg-indigo-300', accent: 'bg-indigo-600', muted: 'bg-slate-200' } as const
export type BarTone = keyof typeof FILL

export interface BarRow { key: string; label: ReactNode; value: number; tone: BarTone; tip?: ReactNode; note?: ReactNode }

/** Horizontal bars from one baseline, value at the tip, a tooltip on hover. */
export function BarChart({ rows, unit = 'tokens', max, labelWidth = 'w-44', caption, format = fmtInt }: {
  rows: BarRow[]; unit?: string; max?: number; labelWidth?: string; caption?: ReactNode; format?: (v: number) => string
}) {
  const [hover, setHover] = useState<string | null>(null)
  const top = max ?? Math.max(1, ...rows.map(r => r.value))
  return (
    <figure>
      <div className="space-y-1.5" role="list">
        {rows.map(r => (
          <div key={r.key} role="listitem" className="relative flex items-center gap-2 text-xs"
            onMouseEnter={() => setHover(r.key)} onMouseLeave={() => setHover(null)}>
            <div className={`${labelWidth} shrink-0 truncate text-slate-600`}>{r.label}</div>
            <div className="relative flex h-6 flex-1 items-center">
              <div className={`h-[18px] rounded-r ${FILL[r.tone]} ${hover && hover !== r.key ? 'opacity-60' : ''}`}
                style={{ width: `${Math.max(0.5, (r.value / top) * 100)}%` }} />
              <span className="ml-1.5 whitespace-nowrap font-mono text-[11px] text-slate-700">{format(r.value)}{r.note && <span className="text-slate-500"> {r.note}</span>}</span>
              {hover === r.key && r.tip && (
                <div className="pointer-events-none absolute left-0 top-full z-10 mt-1 max-w-xs rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] text-slate-700 shadow-lg">
                  {r.tip}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      {caption && <figcaption className="mt-2 text-[11px] text-slate-500">{caption} · values in {unit}</figcaption>}
    </figure>
  )
}

export interface PairRow { key: string; label: ReactNode; a: number | null; b: number | null; tip?: ReactNode }

/** Two bars per row (baseline vs the prompt being shown), with a legend. */
export function PairChart({ rows, aName, bName, caption }: { rows: PairRow[]; aName: string; bName: string; caption?: ReactNode }) {
  const [hover, setHover] = useState<string | null>(null)
  const top = Math.max(1, ...rows.flatMap(r => [r.a ?? 0, r.b ?? 0]))
  const bar = (v: number | null, fill: string) => v == null ? null : (
    <div className="flex h-[11px] items-center">
      <div className={`h-full rounded-r ${fill}`} style={{ width: `${Math.max(0.5, (v / top) * 100)}%` }} />
      <span className="ml-1.5 font-mono text-[10px] leading-none text-slate-700">{fmtInt(v)}</span>
    </div>
  )
  return (
    <figure>
      <div className="mb-2 flex gap-4 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-slate-400" />{aName}</span>
        {bName && <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-indigo-600" />{bName}</span>}
      </div>
      <div className="space-y-2">
        {rows.map(r => (
          <div key={r.key} className="relative flex items-center gap-2 text-xs" onMouseEnter={() => setHover(r.key)} onMouseLeave={() => setHover(null)}>
            <div className="w-48 shrink-0 truncate text-slate-600">{r.label}</div>
            <div className="flex-1 space-y-[2px]">{bar(r.a, 'bg-slate-400')}{bar(r.b, 'bg-indigo-600')}</div>
            {hover === r.key && r.tip && (
              <div className="pointer-events-none absolute left-48 top-full z-10 mt-1 max-w-sm rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] text-slate-700 shadow-lg">{r.tip}</div>
            )}
          </div>
        ))}
      </div>
      {caption && <figcaption className="mt-2 text-[11px] text-slate-500">{caption}</figcaption>}
    </figure>
  )
}

/** Underlined sub-tabs that split a busy panel into views. */
export function SubTabs<T extends string>({ tabs, value, onChange }: {
  tabs: { id: T; label: string; badge?: ReactNode }[]; value: T; onChange: (t: T) => void
}) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 border-b border-slate-200">
      {tabs.map(t => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={`-mb-px flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition ${value === t.id ? 'border-indigo-600 text-indigo-700' : 'border-transparent text-slate-500 hover:text-slate-800'}`}>
          {t.label}{t.badge}
        </button>
      ))}
    </div>
  )
}

export interface RadarAxis { key: string; label: string; value: number }

/** One-series radar on a √ scale (so a 5% share is still visible next to a 70% one); the top axis is marked. */
export function Radar({ axes, highlight, size = 300 }: { axes: RadarAxis[]; highlight?: string; size?: number }) {
  const [hover, setHover] = useState<string | null>(null)
  const c = size / 2, R = size / 2 - 58
  const pt = (i: number, r: number) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / axes.length
    return [c + Math.cos(a) * r, c + Math.sin(a) * r] as const
  }
  const rad = (v: number) => Math.sqrt(Math.max(0, Math.min(1, v))) * R
  const poly = axes.map((ax, i) => pt(i, rad(ax.value)).join(',')).join(' ')
  const rings = [0.05, 0.25, 0.5, 1]
  const hovered = axes.find(a => a.key === hover)
  return (
    <figure className="relative">
      <svg viewBox={`-50 0 ${size + 100} ${size}`} className="mx-auto block w-full max-w-[400px]" role="img"
        aria-label={`Task-type scores: ${axes.map(a => `${a.label} ${(a.value * 100).toFixed(0)}%`).join(', ')}`}>
        {rings.map(f => (
          <polygon key={f} points={axes.map((_, i) => pt(i, rad(f)).join(',')).join(' ')} fill="none" stroke="#e2e8f0" strokeWidth={1} />
        ))}
        {axes.map((_, i) => { const [x, y] = pt(i, R); return <line key={i} x1={c} y1={c} x2={x} y2={y} stroke="#e2e8f0" strokeWidth={1} /> })}
        <polygon points={poly} fill="#4f46e5" fillOpacity={0.1} stroke="#4f46e5" strokeWidth={2} strokeLinejoin="round" />
        {axes.map((ax, i) => {
          const [x, y] = pt(i, rad(ax.value))
          const [lx, ly] = pt(i, R + 16)
          const anchor = Math.abs(lx - c) < 8 ? 'middle' : lx > c ? 'start' : 'end'
          const top = ax.key === highlight
          return (
            <g key={ax.key} onMouseEnter={() => setHover(ax.key)} onMouseLeave={() => setHover(null)} className="cursor-default">
              <circle cx={x} cy={y} r={12} fill="transparent" />
              <circle cx={x} cy={y} r={top ? 5 : 4} fill="#4f46e5" stroke="#fff" strokeWidth={2} />
              <text x={lx} y={ly} textAnchor={anchor} dominantBaseline="middle"
                className={`text-[10px] ${top ? 'fill-slate-900 font-semibold' : 'fill-slate-500'}`}>{ax.label}</text>
            </g>
          )
        })}
      </svg>
      {hovered && (
        <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[11px] text-slate-700 shadow-lg">
          {hovered.label}: <b>{(hovered.value * 100).toFixed(1)}%</b>
        </div>
      )}
      <figcaption className="mt-1 text-center text-[11px] text-slate-500">Share of the task-type score · radius on a √ scale · rings at 5%, 25%, 50%, 100%</figcaption>
    </figure>
  )
}

const SEVERITY = {
  low: { fill: 'bg-emerald-500', text: 'text-emerald-700', icon: '●', label: 'Low' },
  medium: { fill: 'bg-amber-500', text: 'text-amber-700', icon: '▲', label: 'Medium' },
  high: { fill: 'bg-rose-500', text: 'text-rose-700', icon: '■', label: 'High' },
} as const

/** Segmented meter: filled cells carry the severity, the track is the same row unfilled. */
export function SeverityMeter({ level, score, max }: { level: keyof typeof SEVERITY; score: number; max: number }) {
  const s = SEVERITY[level]
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className={`text-sm font-semibold ${s.text}`} aria-hidden>{s.icon}</span>
        <span className="text-sm font-semibold text-slate-900">{s.label} severity</span>
        <span className="font-mono text-xs text-slate-500">{score}/{max}</span>
      </div>
      <div className="mt-1.5 flex gap-[2px]" role="meter" aria-valuenow={score} aria-valuemin={0} aria-valuemax={max} aria-label="Task severity">
        {Array.from({ length: max }, (_, i) => (
          <div key={i} className={`h-2 flex-1 first:rounded-l last:rounded-r ${i < score ? s.fill : 'bg-slate-200'}`} />
        ))}
      </div>
    </div>
  )
}
