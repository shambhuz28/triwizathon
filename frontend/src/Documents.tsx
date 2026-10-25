import { useRef, useState, type Dispatch, type SetStateAction } from 'react'
import type { DocumentInfo, RagMode, UploadResponse } from './types'
import { KIND_STYLE } from './ui'

const ACCEPT = '.pdf,.docx,.pptx,.xlsx,.xlsm,.csv,.tsv,.txt,.md,.json,.jsonl,.html,.htm,.xml,.yaml,.yml,.rst,.log'

const fmtSize = (b: number) => b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`
const PAGE_WORD: Record<string, string> = { pdf: 'page', pptx: 'slide', xlsx: 'sheet', xlsm: 'sheet' }

/** Upload files for S0 and pick when the pipeline may draw on them. Every listed file is attached to the next run. */
export default function Documents({ docs, setDocs, mode, setMode }: {
  docs: DocumentInfo[]; setDocs: Dispatch<SetStateAction<DocumentInfo[]>>; mode: RagMode; setMode: (m: RagMode) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const [errors, setErrors] = useState<string[]>([])

  async function upload(files: FileList | null) {
    const list = Array.from(files ?? [])
    if (!list.length) return
    setBusy(true)
    setErrors([])
    const form = new FormData()
    list.forEach(f => form.append('files', f))
    try {
      const r = await fetch('/api/v1/documents', { method: 'POST', body: form })
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
      const d: UploadResponse = await r.json()
      setDocs(prev => [...prev.filter(p => !d.documents.some(n => n.id === p.id)), ...d.documents])
      setErrors(d.errors.map(e => `${e.filename}: ${e.error}`))
    } catch (e) {
      setErrors([`Upload failed: ${e instanceof Error ? e.message : String(e)}`])
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }

  async function remove(id: string) {
    await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' }).catch(() => {})
    setDocs(prev => prev.filter(d => d.id !== id))
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files) }}
      className={`mt-3 rounded-lg border border-dashed p-3 transition ${drag ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-white'}`}>
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={() => input.current?.click()} disabled={busy}
          className="flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-4 w-4" aria-hidden>
            <path d="M14.5 9.5 9 15a3.5 3.5 0 0 1-5-5l6-6a2.3 2.3 0 0 1 3.3 3.3L7.5 13a1.2 1.2 0 0 1-1.7-1.7L11 6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {busy ? 'Indexing…' : 'Attach documents'}
        </button>
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden" onChange={e => upload(e.target.files)} />
        <span className="text-xs text-slate-500">PDF, Word, PowerPoint, Excel, CSV or text · drop files here · questions about them pull the relevant passages in (S0)</span>
        {docs.length > 0 && (
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-600">Use documents
            <select value={mode} onChange={e => setMode(e.target.value as RagMode)} className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs">
              <option value="auto">when the prompt is about them</option>
              <option value="always">always</option>
              <option value="off">never</option>
            </select>
          </label>
        )}
      </div>
      {docs.length > 0 && (
        <ul className="mt-2.5 flex flex-wrap gap-2">
          {docs.map(d => (
            <li key={d.id} className="flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 py-1 pl-2 pr-1 text-xs">
              <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${KIND_STYLE[d.kind] ?? 'bg-slate-200 text-slate-700'}`}>{d.kind}</span>
              <span className="max-w-56 truncate font-medium text-slate-800" title={d.filename}>{d.filename}</span>
              <span className="text-slate-400">
                {d.pages != null && `${d.pages} ${PAGE_WORD[d.kind] ?? 'part'}${d.pages === 1 ? '' : 's'} · `}{d.chunks} chunk{d.chunks === 1 ? '' : 's'} · {fmtSize(d.size)}
              </span>
              <button onClick={() => remove(d.id)} title="Remove this document"
                className="rounded px-1.5 text-slate-400 hover:bg-slate-200 hover:text-slate-700">×</button>
            </li>
          ))}
        </ul>
      )}
      {errors.map(e => <p key={e} className="mt-2 rounded-md bg-rose-50 px-2 py-1 text-xs text-rose-700">{e}</p>)}
    </div>
  )
}
