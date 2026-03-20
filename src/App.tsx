/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { flushSync } from 'react-dom';
import axios from 'axios';

type UploadMode = 'knowledge' | 'images' | 'image-query' | 'all-images' | 'all-documents' | 'advanced-rag' | 'evaluation';
type RerankStrategy = 'llm_based' | 'cross_encoder' | 'none';
type ChunkingStrategy = 'simple' | 'semantic' | 'parent-child';
type ImageData = { id: string; s3Url: string; description?: string; score?: number; keywords?: string[] };
type DocumentData = { id: string; text: string; createdAt?: string; model?: string };
type Citation = { id: string; documentId: string; text: string };
type SourceDoc = { id: string; text: string; score?: number; metadata?: Record<string, any> };
type StreamMeta = { citations: Citation[]; images: string[]; confidence?: number; relevantChunks: number; queryType?: 'entity' | 'factual' | 'wide'; queryConfidence?: number; generationParams?: any; sources?: SourceDoc[] };
type StreamChunkEvent =
  | { event: 'metadata';   metadata: Partial<StreamMeta> & { conversationContext?: boolean } }
  | { event: 'sources';    sources: SourceDoc[] }
  | { event: 'token';      token: string }
  | { event: 'citations';  citations: Citation[] }
  | { event: 'correction'; correctedAnswer: string; reason: 'hallucination' }
  | { event: 'done';       metadata: Partial<StreamMeta> }
  | { event: 'error';      error: string };
type ChatMessage = { role: 'user' | 'assistant'; content: string; meta?: StreamMeta | null; isCorrected?: boolean; timestamp: number };
type Chat = { sessionId: string; firstMessage: string; lastActivity: string | Date; turnCount: number; messages?: ChatMessage[] };
type Toast = { id: number; text: string; type: 'ok' | 'err' | 'info' };

const API = import.meta.env.VITE_API_URL;

const fmtTime = (ts: number) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const fmtDate = (d: string | Date) => {
  const date = new Date(d), now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 86400000);
  if (diff === 0) return 'Today'; if (diff === 1) return 'Yesterday'; if (diff < 7) return 'This week';
  return date.toLocaleDateString([], { month: 'short', year: 'numeric' });
};

let toastId = 0;

// ─── Tag ──────────────────────────────────────────────────────
const Tag: React.FC<{ color?: 'green'|'orange'|'red'|'blue'; children: React.ReactNode }> = ({ color, children }) => {
  const base = 'inline-flex items-center gap-1 font-mono text-[0.6rem] tracking-[.05em] px-[7px] py-[2px] rounded border';
  const styles = {
    green:  'border-[color-mix(in_srgb,var(--color-green)_30%,transparent)] text-green bg-[color-mix(in_srgb,var(--color-green)_8%,transparent)]',
    orange: 'border-[color-mix(in_srgb,var(--color-accent)_30%,transparent)] text-accent bg-[color-mix(in_srgb,var(--color-accent)_13%,transparent)]',
    red:    'border-[color-mix(in_srgb,var(--color-red)_30%,transparent)] text-red bg-[color-mix(in_srgb,var(--color-red)_8%,transparent)]',
    blue:   'border-[color-mix(in_srgb,var(--color-accent)_30%,transparent)] text-accent bg-[color-mix(in_srgb,var(--color-accent)_13%,transparent)]',
  };
  return <span className={`${base} ${color ? styles[color] : 'border-border2 text-muted'}`}>{children}</span>;
};

// ─── Toast ────────────────────────────────────────────────────
const ToastContainer: React.FC<{ toasts: Toast[]; onRemove: (id: number) => void }> = ({ toasts, onRemove }) => (
  <div className="fixed bottom-6 right-6 flex flex-col gap-2 z-[9998] pointer-events-none">
    {toasts.map(t => (
      <div key={t.id} className={[
        'flex items-center gap-2 px-3.5 py-2.5 rounded-xl font-mono text-xs pointer-events-auto min-w-[200px] max-w-[340px] shadow-[0_4px_20px_rgba(0,0,0,.35)] backdrop-blur-sm animate-toast-in',
        t.type === 'ok'  ? 'bg-[color-mix(in_srgb,var(--color-green)_12%,transparent)] border border-[color-mix(in_srgb,var(--color-green)_25%,transparent)] text-green' :
        t.type === 'err' ? 'bg-[color-mix(in_srgb,var(--color-red)_12%,transparent)] border border-[color-mix(in_srgb,var(--color-red)_25%,transparent)] text-red' :
                           'bg-[color-mix(in_srgb,var(--color-accent)_12%,transparent)] border border-[color-mix(in_srgb,var(--color-accent)_25%,transparent)] text-accent',
      ].join(' ')}>
        <span>{t.type === 'ok' ? '✓' : t.type === 'err' ? '✗' : '·'}</span>
        <span className="flex-1">{t.text}</span>
        <span className="ml-auto opacity-60 hover:opacity-100 cursor-pointer" onClick={() => onRemove(t.id)}>✕</span>
      </div>
    ))}
  </div>
);

// ─── Skeleton ─────────────────────────────────────────────────
const SidebarSkeleton = () => (
  <div className="p-2 flex flex-col gap-1.5">
    {[80, 60, 90, 55, 70].map((w, i) => (
      <div key={i} className="flex items-center gap-2 px-1.5 py-2">
        <div className="h-3 rounded bg-[linear-gradient(90deg,var(--color-surface2)_25%,var(--color-border)_50%,var(--color-surface2)_75%)] bg-[length:200%_100%] animate-shimmer" style={{ width: `${w}%` }} />
      </div>
    ))}
  </div>
);

// ─── EmptyChats ───────────────────────────────────────────────
const EmptyChats = () => (
  <div className="flex flex-col items-center px-4 py-7 gap-2 opacity-60">
    <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
      <rect x="3" y="7" width="32" height="22" rx="5" stroke="currentColor" strokeWidth="1.8" className="text-muted"/>
      <path d="M10 17h18M10 22h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="text-muted"/>
      <path d="M13 29l-4 5M25 29l4 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" className="text-muted"/>
    </svg>
    <span className="font-sans text-[0.74rem] text-dim text-center leading-relaxed">No chats yet.<br/>Start a new one!</span>
  </div>
);

// ─── Toggle ───────────────────────────────────────────────────
const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; label: string; sub?: string }> = ({ checked, onChange, label, sub }) => (
  <div onClick={() => onChange(!checked)} className="flex items-start gap-2.5 cursor-pointer select-none py-1">
    <div className={`flex-shrink-0 w-8 h-[18px] rounded-full border relative transition-all duration-200 mt-0.5 ${checked ? 'bg-accent border-accent' : 'bg-transparent border-border2'}`}>
      <div className={`absolute top-0.5 w-3 h-3 rounded-full transition-[left] duration-200 ${checked ? 'left-4 bg-white' : 'left-0.5 bg-border2'}`} />
    </div>
    <div>
      <div className={`text-sm transition-colors duration-200 ${checked ? 'text-text' : 'text-muted'}`}>{label}</div>
      {sub && <div className="font-mono text-[0.66rem] text-dim mt-0.5">{sub}</div>}
    </div>
  </div>
);

// ─── Btn ──────────────────────────────────────────────────────
const Btn: React.FC<{ onClick: () => void; disabled?: boolean; danger?: boolean; children: React.ReactNode; accent?: boolean }> = ({ onClick, disabled, danger, children, accent }) => (
  <button onClick={onClick} disabled={disabled} className={[
    'w-full py-2.5 px-5 rounded-lg font-sans font-medium text-sm transition-all duration-150',
    disabled ? 'border border-border2 bg-surface text-dim cursor-not-allowed' :
    danger   ? 'border border-[color-mix(in_srgb,var(--color-red)_25%,transparent)] bg-[color-mix(in_srgb,var(--color-red)_8%,transparent)] text-red cursor-pointer hover:bg-[color-mix(in_srgb,var(--color-red)_15%,transparent)]' :
    accent   ? 'border border-accent-bd bg-accent-bg text-accent cursor-pointer hover:bg-accent hover:text-white' :
               'border border-border2 bg-surface text-muted cursor-pointer hover:text-text',
  ].join(' ')}>{children}</button>
);

// ─── RangeField ───────────────────────────────────────────────
const RangeField: React.FC<{ label: string; value: number | undefined; onChange: (v: number | undefined) => void; min: number; max: number; step: number; fmt?: (v: number) => string; placeholder?: string }> = ({ label, value, onChange, min, max, step, fmt, placeholder = 'auto' }) => {
  const pct = value !== undefined ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <div className="mb-3.5">
      <div className="flex justify-between mb-1.5">
        <span className="font-mono text-[0.6rem] tracking-[.1em] uppercase text-muted">{label}</span>
        <div className="flex gap-1.5 items-center">
          <span className={`font-mono text-[0.72rem] ${value !== undefined ? 'text-accent' : 'text-dim'}`}>{value !== undefined ? (fmt ? fmt(value) : value) : placeholder}</span>
          {value !== undefined && <span onClick={() => onChange(undefined)} className="font-mono text-[0.6rem] text-muted cursor-pointer px-1 border border-border2 rounded-[3px]">✕</span>}
        </div>
      </div>
      <input type="range" min={min} max={max} step={step} value={value ?? min} style={{ '--pct': `${pct}%` } as any} onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
};

// ─── Spin ─────────────────────────────────────────────────────
const Spin: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <span style={{ width: size, height: size }} className="border-2 border-border2 border-t-accent rounded-full inline-block animate-spin flex-shrink-0" />
);

// ─── SendButton ───────────────────────────────────────────────
const SendButton: React.FC<{ active: boolean; isStreaming: boolean; onSend: () => void; onStop: () => void }> = ({ active, isStreaming, onSend, onStop }) => {
  const [firing, setFiring] = useState(false);
  const handleClick = () => {
    if (isStreaming) { onStop(); return; }
    if (!active) return;
    setFiring(true); setTimeout(() => setFiring(false), 420); onSend();
  };
  if (isStreaming) return (
    <button onClick={handleClick} className="w-9 h-9 rounded-[10px] flex items-center justify-center cursor-pointer flex-shrink-0 self-end bg-surface2 border border-border2 hover:bg-border2 transition-colors">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="2.5" y="2.5" width="9" height="9" rx="2" fill="currentColor" className="text-muted"/></svg>
    </button>
  );
  return (
    <button onClick={handleClick} disabled={!active} className={[
      'w-9 h-9 rounded-[10px] border-none flex items-center justify-center flex-shrink-0 self-end transition-all duration-150 overflow-hidden',
      active ? 'bg-accent cursor-pointer shadow-[0_2px_8px_color-mix(in_srgb,var(--color-accent)_40%,transparent)] hover:bg-accent-h hover:scale-105' : 'bg-transparent cursor-not-allowed border border-border2',
      firing ? 'animate-send-pop' : '',
    ].join(' ')}>
      <span className={`mat-icon ${firing ? 'animate-arrow-shoot' : ''}`} style={{ color: active ? '#fff' : 'var(--color-dim)' }}>arrow_forward</span>
    </button>
  );
};

// ─── ChatSidebar ──────────────────────────────────────────────
const ChatSidebar: React.FC<{
  chats: Chat[]; activeChatId: string | null;
  onSelect: (id: string) => void; onNew: () => void;
  onDelete: (id: string) => void; onRename: (id: string, name: string) => void;
  loading?: boolean;
}> = ({ chats, activeChatId, onSelect, onNew, onDelete, onRename, loading }) => {
  const [search, setSearch] = useState('');
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState('');
  const filtered = search.trim() ? chats.filter(c => (c.firstMessage || '').toLowerCase().includes(search.toLowerCase())) : chats;
  const groups: { label: string; chats: Chat[] }[] = [];
  filtered.forEach(chat => { const label = fmtDate(chat.lastActivity); const g = groups.find(g => g.label === label); if (g) g.chats.push(chat); else groups.push({ label, chats: [chat] }); });
  const startRename = (chat: Chat, e: React.MouseEvent) => { e.stopPropagation(); setRenamingId(chat.sessionId); setRenameVal(chat.firstMessage || ''); };
  const commitRename = (id: string) => { if (renameVal.trim()) onRename(id, renameVal.trim()); setRenamingId(null); };
  return (
    <div className="w-[244px] bg-surface border-r border-border flex flex-col flex-shrink-0 h-full overflow-hidden">
      <div className="px-2.5 pt-3 pb-2.5 border-b border-border flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center font-mono text-[0.58rem] font-bold text-white flex-shrink-0">R</div>
          <div>
            <div className="font-sans text-[0.84rem] font-bold text-text">RAG System</div>
            <div className="font-mono text-[0.52rem] text-dim tracking-[.08em]">knowledge base</div>
          </div>
        </div>
        <button onClick={onNew} className="w-full py-[0.45rem] rounded-lg border border-accent-bd bg-accent-bg text-accent font-sans text-[0.75rem] font-semibold flex items-center justify-center gap-1.5 transition-all duration-150 hover:bg-accent hover:text-white cursor-pointer">
          <svg width="11" height="11" viewBox="0 0 12 12" fill="none"><path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
          New Chat
        </button>
        <input
          className="bg-surface2 border border-border rounded-lg py-[5px] px-2.5 text-[0.78rem] font-sans text-text outline-none w-full placeholder:text-dim transition-all duration-150 focus:border-accent focus:shadow-[0_0_0_2px_var(--color-accent-bg)]"
          placeholder="Search chats…" value={search} onChange={e => setSearch(e.target.value)}
        />
      </div>
      <div className="flex-1 overflow-y-auto py-1.5 px-2">
        {loading ? <SidebarSkeleton /> : filtered.length === 0
          ? (search ? <div className="py-4 px-2.5 font-sans text-[0.74rem] text-dim text-center">No results</div> : <EmptyChats />)
          : groups.map(({ label, chats: gc }) => (
            <div key={label}>
              <div className="font-mono text-[0.6rem] tracking-[.1em] uppercase text-dim px-1.5 pt-2.5 pb-1">{label}</div>
              {gc.map(chat => (
                <div key={chat.sessionId}
                  onClick={() => renamingId !== chat.sessionId && onSelect(chat.sessionId)}
                  onDoubleClick={e => startRename(chat, e)}
                  className={`px-2.5 py-[7px] cursor-pointer flex items-center gap-2 transition-all duration-150 border-l-2 rounded-lg mb-0.5 group ${activeChatId === chat.sessionId ? 'bg-accent-bg border-l-accent' : 'border-l-transparent hover:bg-surface2'}`}>
                  {renamingId === chat.sessionId
                    ? <input autoFocus value={renameVal} onChange={e => setRenameVal(e.target.value)}
                        onBlur={() => commitRename(chat.sessionId)}
                        onKeyDown={e => { if (e.key === 'Enter') commitRename(chat.sessionId); if (e.key === 'Escape') setRenamingId(null); }}
                        onClick={e => e.stopPropagation()}
                        className="flex-1 bg-surface border border-accent rounded-[5px] text-text font-sans text-[0.78rem] py-0.5 px-1.5 outline-none min-w-0"
                      />
                    : <div title={chat.firstMessage} className={`font-sans text-[0.78rem] overflow-hidden text-ellipsis whitespace-nowrap flex-1 min-w-0 ${activeChatId === chat.sessionId ? 'text-text font-medium' : 'text-muted'}`}>
                        {chat.firstMessage || chat.sessionId.slice(0, 14)}
                      </div>
                  }
                  {renamingId !== chat.sessionId && (
                    <span onClick={e => { e.stopPropagation(); onDelete(chat.sessionId); }}
                      className="opacity-0 group-hover:opacity-100 font-mono text-[0.65rem] text-red cursor-pointer py-[2px] px-[5px] rounded border border-[color-mix(in_srgb,var(--color-red)_25%,transparent)] bg-[color-mix(in_srgb,var(--color-red)_8%,transparent)] flex-shrink-0 transition-opacity duration-150">✕</span>
                  )}
                </div>
              ))}
            </div>
          ))
        }
      </div>
      <div className="px-3 py-2 border-t border-border">
        <div className="font-mono text-[0.58rem] text-dim">{chats.length} chat{chats.length !== 1 ? 's' : ''}</div>
      </div>
    </div>
  );
};

const MarkdownText: React.FC<{ text: string }> = ({ text }) => {
  const render = (src: string): React.ReactNode[] => {
    const lines = src.split('\n'); const nodes: React.ReactNode[] = []; let i = 0;
    const inlineRender = (s: string, key: string): React.ReactNode => {
      const parts: React.ReactNode[] = []; let buf = ''; let j = 0; let k = 0;
      while (j < s.length) {
        if (s[j]==='`'){const end=s.indexOf('`',j+1);if(end!==-1){if(buf){parts.push(<span key={k++}>{buf}</span>);buf='';}parts.push(<code key={k++} className="font-mono text-[0.82rem] bg-surface2 border border-border2 rounded-[5px] px-1.5 py-px text-accent">{s.slice(j+1,end)}</code>);j=end+1;continue;}}
        if(s[j]==='*'&&s[j+1]==='*'){const end=s.indexOf('**',j+2);if(end!==-1){if(buf){parts.push(<span key={k++}>{buf}</span>);buf='';}parts.push(<strong key={k++} className="font-bold text-text">{s.slice(j+2,end)}</strong>);j=end+2;continue;}}
        if(s[j]==='*'&&s[j+1]!=='*'){const end=s.indexOf('*',j+1);if(end!==-1){if(buf){parts.push(<span key={k++}>{buf}</span>);buf='';}parts.push(<em key={k++} className="italic text-muted">{s.slice(j+1,end)}</em>);j=end+1;continue;}}
        if(s[j]==='['){const te=s.indexOf(']',j+1);if(te!==-1&&s[te+1]==='('){const ue=s.indexOf(')',te+2);if(ue!==-1){if(buf){parts.push(<span key={k++}>{buf}</span>);buf='';}parts.push(<a key={k++} href={s.slice(te+2,ue)} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-[3px] decoration-[color-mix(in_srgb,var(--color-accent)_35%,transparent)] break-all hover:opacity-80">{s.slice(j+1,te)}</a>);j=ue+1;continue;}}}
        if(s.slice(j,j+8)==='https://'||s.slice(j,j+7)==='http://'){if(buf){parts.push(<span key={k++}>{buf}</span>);buf='';}let ue=j;while(ue<s.length&&!/[\s,)>\]"']/.test(s[ue]))ue++;while(ue>j&&/[.,!?;:]/.test(s[ue-1]))ue--;const href=s.slice(j,ue);parts.push(<a key={k++} href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-[3px] break-all hover:opacity-80">{href}</a>);j=ue;continue;}
        buf+=s[j++];
      }
      if(buf)parts.push(<span key={k++}>{buf}</span>);
      return <React.Fragment key={key}>{parts}</React.Fragment>;
    };
    while(i<lines.length){
      const line=lines[i];
      if(line.trimStart().startsWith('```')){const cl:string[]=[];i++;while(i<lines.length&&!lines[i].trimStart().startsWith('```')){cl.push(lines[i]);i++;}nodes.push(<pre key={i} className="bg-surface2 border border-border rounded-[10px] p-3.5 my-2.5 overflow-x-auto"><code className="font-mono text-[0.82rem] leading-[1.7] text-text">{cl.join('\n')}</code></pre>);i++;continue;}
      const h3=line.match(/^###\s+(.+)/),h2=line.match(/^##\s+(.+)/),h1=line.match(/^#\s+(.+)/);
      if(h3){nodes.push(<h3 key={i} className="text-[0.97rem] font-semibold text-text mt-3 mb-1 first:mt-0">{inlineRender(h3[1],`h${i}`)}</h3>);i++;continue;}
      if(h2){nodes.push(<h2 key={i} className="text-[1.1rem] font-bold text-text mt-4 mb-1.5 first:mt-0">{inlineRender(h2[1],`h${i}`)}</h2>);i++;continue;}
      if(h1){nodes.push(<h1 key={i} className="text-[1.35rem] font-bold text-text mt-4 mb-2 first:mt-0">{inlineRender(h1[1],`h${i}`)}</h1>);i++;continue;}
      if(/^---+$/.test(line.trim())){nodes.push(<hr key={i} className="border-none border-t border-border my-3.5"/>);i++;continue;}
      if(line.startsWith('>')){nodes.push(<blockquote key={i} className="border-l-[3px] border-l-accent pl-3.5 py-1 my-2 text-muted italic">{inlineRender(line.replace(/^>\s?/,''),`bq${i}`)}</blockquote>);i++;continue;}
      if(/^[-*+]\s/.test(line)){const items:React.ReactNode[]=[];while(i<lines.length&&/^[-*+]\s/.test(lines[i])){items.push(<li key={i} className="list-disc leading-[1.7]">{inlineRender(lines[i].replace(/^[-*+]\s/,''),`li${i}`)}</li>);i++;}nodes.push(<ul key={`ul${i}`} className="pl-5 my-1.5 flex flex-col gap-[3px]">{items}</ul>);continue;}
      if(/^\d+\.\s/.test(line)){const items:React.ReactNode[]=[];while(i<lines.length&&/^\d+\.\s/.test(lines[i])){items.push(<li key={i} className="list-decimal leading-[1.7]">{inlineRender(lines[i].replace(/^\d+\.\s/,''),`oli${i}`)}</li>);i++;}nodes.push(<ol key={`ol${i}`} className="pl-5 my-1.5 flex flex-col gap-[3px]">{items}</ol>);continue;}
      if(line.trim()===''){i++;continue;}
      nodes.push(<p key={i} className="my-1.5 first:mt-0">{inlineRender(line,`p${i}`)}</p>);i++;
    }
    return nodes;
  };
  return <div className="font-sans text-[0.93rem] leading-[1.85] text-text">{render(text)}</div>;
};

const MessageBubble: React.FC<{ msg: ChatMessage; onLightbox: (url: string) => void }> = ({ msg, onLightbox }) => {
  const [expandedSrc, setExpandedSrc] = useState(false);
  const isUser = msg.role === 'user';
  return (
    <div className={`flex flex-col animate-fade-up ${isUser ? 'items-end py-1' : 'items-start py-1'}`}>
      {!isUser && <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center font-mono text-[0.58rem] font-bold text-white flex-shrink-0 mb-1.5">R</div>}
      <div className={isUser ? 'max-w-[72%] bg-user-msg border border-accent-bd rounded-[18px_18px_4px_18px] px-[18px] py-3' : 'max-w-[86%]'}>
        {isUser ? <div className="font-sans text-[0.93rem] leading-[1.85] text-text whitespace-pre-wrap">{msg.content}</div> : <MarkdownText text={msg.content} />}
        {msg.meta?.images && msg.meta.images.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border">
            <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Images ({msg.meta.images.length})</div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2 mt-2.5">
              {msg.meta.images.map((url, i) => (
                <div key={i} onClick={() => onLightbox(url)} className="rounded-lg overflow-hidden border border-border2 cursor-pointer hover:border-accent hover:scale-[1.02] transition-all duration-150">
                  <img src={url} alt="" className="w-full h-[90px] object-cover block" onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                </div>
              ))}
            </div>
          </div>
        )}
        {msg.meta?.citations && msg.meta.citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border">
            <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Citations ({msg.meta.citations.length})</div>
            <div className="flex flex-col gap-1.5">
              {msg.meta.citations.slice(0, 4).map((c, i) => (
                <div key={i} className="flex gap-2">
                  <span className="font-mono text-[0.62rem] text-accent flex-shrink-0">[{i+1}]</span>
                  <div className="font-mono text-[0.7rem] text-muted leading-relaxed">{c.text.slice(0, 120)}{c.text.length > 120 ? '…' : ''}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {msg.meta?.sources && msg.meta.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border">
            <div className="flex justify-between items-center mb-2">
              <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim">Sources ({msg.meta.sources.length})</div>
              <span onClick={() => setExpandedSrc(v => !v)} className="font-mono text-[0.6rem] text-muted cursor-pointer py-px px-[7px] border border-border2 rounded">{expandedSrc ? 'collapse' : 'expand'}</span>
            </div>
            {expandedSrc && msg.meta.sources.map((src, i) => (
              <div key={i} className="bg-surface2 border border-border rounded-lg p-2.5 mb-1.5">
                <div className="flex justify-between mb-1">
                  <span className="font-mono text-[0.58rem] text-muted">#{i+1} · {src.id.slice(0,8)}…</span>
                  {src.score !== undefined && <span className="font-mono text-[0.58rem] text-muted">{src.score.toFixed(3)}</span>}
                </div>
                <p className="font-mono text-[0.72rem] text-muted leading-relaxed m-0">{src.text.slice(0, 240)}{src.text.length > 240 ? '…' : ''}</p>
              </div>
            ))}
          </div>
        )}
        {(msg.isCorrected || msg.meta?.queryType || msg.meta?.confidence !== undefined || msg.meta?.relevantChunks !== undefined) && (
          <div className="flex gap-1 flex-wrap mt-2.5">
            {msg.isCorrected && <Tag color="red">hallucination filtered</Tag>}
            {msg.meta?.queryType && <Tag color={msg.meta.queryType === 'entity' ? 'blue' : msg.meta.queryType === 'wide' ? 'orange' : 'green'}>{msg.meta.queryType}</Tag>}
            {msg.meta?.confidence !== undefined && <Tag color="green">{(msg.meta.confidence*100).toFixed(0)}% conf</Tag>}
            {msg.meta?.relevantChunks !== undefined && <Tag>{msg.meta.relevantChunks} chunks</Tag>}
          </div>
        )}
      </div>
      <div className="font-mono text-[0.58rem] text-dim mt-[3px] text-right">{fmtTime(msg.timestamp)}</div>
    </div>
  );
};

// ─── ChatInput ────────────────────────────────────────────────
const ChatInput: React.FC<{ question: string; setQuestion: (v: string) => void; onSend: () => void; onStop: () => void; isStreaming: boolean; hasMessages: boolean; textareaRef: React.RefObject<HTMLTextAreaElement> }> = ({ question, setQuestion, onSend, onStop, isStreaming, hasMessages, textareaRef }) => {
  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!isStreaming && question.trim()) onSend(); } };
  if (!hasMessages) return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-10 pointer-events-none animate-fade-in">
      <div className="pointer-events-auto w-[min(640px,92%)] flex flex-col items-center">
        <div className="text-center mb-6 animate-fade-up">
          <div className="w-[52px] h-[52px] rounded-[14px] bg-accent flex items-center justify-center font-mono text-[1.1rem] font-bold text-white mx-auto mb-3.5 shadow-[0_8px_24px_var(--color-accent-bg)]">R</div>
          <div className="font-sans text-[1.35rem] font-bold text-text mb-1.5">RAG System</div>
          <div className="font-sans text-[0.88rem] text-muted">Ask anything about your knowledge base</div>
        </div>
        <div className="w-full flex items-end bg-surface border border-border2 rounded-[18px] px-5 py-3.5 shadow-[0_4px_28px_rgba(0,0,0,.12)] focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--color-accent-bg)] transition-all duration-200">
          <textarea ref={textareaRef} value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={handleKey} placeholder="What would you like to know?" rows={1}
            className="flex-1 resize-none min-h-[26px] max-h-[200px] overflow-y-auto leading-[1.6] bg-transparent border-none outline-none text-[1rem] font-sans text-text placeholder:text-dim mb-[5px]" />
          <SendButton active={!!question.trim()} isStreaming={isStreaming} onSend={onSend} onStop={onStop} />
        </div>
        <div className="font-mono text-[0.6rem] text-dim mt-2 tracking-[.04em]">Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  );
  return (
    <div className="px-5 pt-2.5 pb-4 bg-bg flex-shrink-0">
      <div className="flex items-end bg-surface border border-border2 rounded-[16px] px-2.5 py-2.5 pl-4 focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--color-accent-bg)] transition-all duration-200">
        <textarea ref={textareaRef} value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={handleKey} placeholder="Message RAG System…" rows={1}
          className="flex-1 resize-none min-h-[26px] max-h-[200px] overflow-y-auto leading-[1.6] bg-transparent border-none outline-none text-[0.93rem] font-sans text-text placeholder:text-dim mb-1" />
        <SendButton active={!!question.trim()} isStreaming={isStreaming} onSend={onSend} onStop={onStop} />
      </div>
      <div className="text-center font-mono text-[0.58rem] text-dim mt-[7px] tracking-[.04em]">Enter to send · Shift+Enter for new line</div>
    </div>
  );
};

// ─── PipelinePanel ────────────────────────────────────────────
interface PipelineProps {
  useHybridSearch:boolean;setUseHybridSearch:(v:boolean)=>void;
  useQueryTransformation:boolean;setUseQueryTransformation:(v:boolean)=>void;
  useContextualCompression:boolean;setUseContextualCompression:(v:boolean)=>void;
  useCitationTracking:boolean;setUseCitationTracking:(v:boolean)=>void;
  useConversationMemory:boolean;setUseConversationMemory:(v:boolean)=>void;
  useKnowledgeGraph:boolean;setUseKnowledgeGraph:(v:boolean)=>void;
  includeSources:boolean;setIncludeSources:(v:boolean)=>void;
  useReranking:boolean;setUseReranking:(v:boolean)=>void;
  rerankStrategy:RerankStrategy;setRerankStrategy:(v:RerankStrategy)=>void;
  limit:number|undefined;setLimit:(v:number|undefined)=>void;
  scoreThreshold:number|undefined;setScoreThreshold:(v:number|undefined)=>void;
  temperature:number|undefined;setTemperature:(v:number|undefined)=>void;
  topP:number|undefined;setTopP:(v:number|undefined)=>void;
  topK:number|undefined;setTopK:(v:number|undefined)=>void;
  maxTokens:number|undefined;setMaxTokens:(v:number|undefined)=>void;
}
const PipelinePanel: React.FC<PipelineProps> = (p) => (
  <div className="bg-surface border border-border rounded-xl p-[18px]">
    <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Pipeline</div>
    <div className="grid grid-cols-2 gap-x-4">
      <Toggle checked={p.useHybridSearch} onChange={p.setUseHybridSearch} label="Hybrid Search" sub="Vector + BM25" />
      <Toggle checked={p.useQueryTransformation} onChange={p.setUseQueryTransformation} label="Query Expansion" sub="Rephrase + expand" />
      <Toggle checked={p.useContextualCompression} onChange={p.setUseContextualCompression} label="Compression" sub="Extract relevant" />
      <Toggle checked={p.useCitationTracking} onChange={p.setUseCitationTracking} label="Citations" sub="Track sources" />
      <Toggle checked={p.useConversationMemory} onChange={p.setUseConversationMemory} label="Memory" sub="Session history" />
      <Toggle checked={p.useKnowledgeGraph} onChange={p.setUseKnowledgeGraph} label="Knowledge Graph" sub="Neo4j enrichment" />
      <Toggle checked={p.includeSources} onChange={p.setIncludeSources} label="Return Sources" sub="Attach chunks" />
    </div>
    <div className="border-t border-border mt-3.5 pt-3.5">
      <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Re-ranking</div>
      <Toggle checked={p.useReranking} onChange={p.setUseReranking} label="Enable Re-ranking" />
      {p.useReranking && <div className="mt-2.5"><div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Strategy</div>
        <select value={p.rerankStrategy} onChange={e => p.setRerankStrategy(e.target.value as RerankStrategy)}>
          <option value="none">Hybrid (default)</option><option value="cross_encoder">Cross-encoder (listwise)</option><option value="llm_based">LLM-based</option>
        </select>
      </div>}
    </div>
    <div className="border-t border-border mt-3.5 pt-3.5">
      <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">Retrieval</div>
      <RangeField label="Chunks" value={p.limit} onChange={p.setLimit} min={1} max={20} step={1} placeholder="6" />
      <RangeField label="Score threshold" value={p.scoreThreshold} onChange={p.setScoreThreshold} min={0} max={1} step={0.05} fmt={v=>v.toFixed(2)} placeholder="off" />
    </div>
    <div className="border-t border-border mt-3.5 pt-3.5">
      <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">LLM Generation</div>
      <RangeField label="Temperature" value={p.temperature} onChange={p.setTemperature} min={0} max={1} step={0.05} fmt={v=>v.toFixed(2)} placeholder="auto" />
      <RangeField label="Top-p" value={p.topP} onChange={p.setTopP} min={0} max={1} step={0.05} fmt={v=>v.toFixed(2)} />
      <RangeField label="Top-k" value={p.topK} onChange={p.setTopK} min={1} max={100} step={1} />
      <RangeField label="Max tokens" value={p.maxTokens} onChange={p.setMaxTokens} min={100} max={8192} step={100} placeholder="auto" />
    </div>
  </div>
);

// ─── Card / Label ─────────────────────────────────────────────
const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <div className={`bg-surface border border-border rounded-xl p-[18px] ${className}`}>{children}</div>
);
const Label = ({ children }: { children: React.ReactNode }) => (
  <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-2">{children}</div>
);

// ─── Main ─────────────────────────────────────────────────────
const RagDemo: React.FC = () => {
  const [mode, setMode] = useState<UploadMode>('advanced-rag');
  const [darkMode, setDarkMode] = useState(true);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatsLoading, setChatsLoading] = useState(true);
  const [streamText, setStreamText] = useState('');
  const [streamMeta, setStreamMeta] = useState<StreamMeta | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isCorrected, setIsCorrected] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const accumulatedRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [question, setQuestion] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [images, setImages] = useState<File[]>([]);
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>('simple');
  const [enableKnowledgeGraph, setEnableKnowledgeGraph] = useState(false);
  const [busy, setBusy] = useState(false);
  const [useHybridSearch, setUseHybridSearch] = useState(true);
  const [useReranking, setUseReranking] = useState(true);
  const [rerankStrategy, setRerankStrategy] = useState<RerankStrategy>('none');
  const [useQueryTransformation, setUseQueryTransformation] = useState(true);
  const [useContextualCompression, setUseContextualCompression] = useState(false);
  const [useConversationMemory, setUseConversationMemory] = useState(true);
  const [useCitationTracking, setUseCitationTracking] = useState(true);
  const [useKnowledgeGraph, setUseKnowledgeGraph] = useState(false);
  const [includeSources, setIncludeSources] = useState(false);
  const [limit, setLimit] = useState<number | undefined>(undefined);
  const [scoreThreshold, setScoreThreshold] = useState<number | undefined>(undefined);
  const [temperature, setTemperature] = useState<number | undefined>(undefined);
  const [topP, setTopP] = useState<number | undefined>(undefined);
  const [topK, setTopK] = useState<number | undefined>(undefined);
  const [maxTokens, setMaxTokens] = useState<number | undefined>(undefined);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [retrievedImages, setRetrievedImages] = useState<ImageData[]>([]);
  const [allDocuments, setAllDocuments] = useState<DocumentData[]>([]);
  const [evalResults, setEvalResults] = useState<any>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [lightboxImg, setLightboxImg] = useState<string | null>(null);

  useEffect(() => {
    if (darkMode) document.documentElement.classList.remove('light');
    else document.documentElement.classList.add('light');
  }, [darkMode]);

  const addToast = useCallback((text: string, type: Toast['type']) => {
    const id = ++toastId; setToasts(prev => [...prev, { id, text, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);
  const ok = useCallback((t: string) => addToast(t, 'ok'), [addToast]);
  const err = useCallback((t: string) => addToast(t, 'err'), [addToast]);
  const inf = useCallback((t: string) => addToast(t, 'info'), [addToast]);
  const removeToast = (id: number) => setToasts(prev => prev.filter(t => t.id !== id));

  const activeChat = chats.find(c => c.sessionId === activeChatId) ?? null;
  const hasMessages = (activeChat?.messages?.length ?? 0) > 0 || isStreaming;

  useEffect(() => {
    const el = messagesContainerRef.current; if (!el) return;
    const h = () => setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 120);
    el.addEventListener('scroll', h); return () => el.removeEventListener('scroll', h);
  }, []);
  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  const fetchChats = useCallback(async () => {
    try { setChatsLoading(true); const r = await axios.get(`${API}/rag/chats`); setChats(r.data.data ?? []); }
    catch { /* silent */ } finally { setChatsLoading(false); }
  }, []);
  useEffect(() => { fetchChats(); }, [fetchChats]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [streamText, activeChat?.messages?.length]);
  useEffect(() => {
    if (textareaRef.current) { textareaRef.current.style.height = 'auto'; textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px'; }
  }, [question]);

  const createNewChat = useCallback(() => {
    const sid = `session_${Date.now()}`;
    setChats(prev => [{ sessionId: sid, firstMessage: 'New Chat', lastActivity: new Date(), turnCount: 0, messages: [] }, ...prev]);
    setActiveChatId(sid); setQuestion(''); setStreamText(''); setStreamMeta(null); setIsCorrected(false);
    if (mode !== 'advanced-rag') setMode('advanced-rag');
    setTimeout(() => textareaRef.current?.focus(), 80);
  }, [mode]);

  const deleteChat = useCallback(async (sid: string) => {
    try { await axios.delete(`${API}/rag/chats/${sid}`); setChats(prev => { const next = prev.filter(c => c.sessionId !== sid); if (activeChatId === sid) setActiveChatId(next.length ? next[0].sessionId : null); return next; }); }
    catch (e: any) { err(e.message); }
  }, [activeChatId]);

  const renameChat = useCallback((sid: string, name: string) => { setChats(prev => prev.map(c => c.sessionId === sid ? { ...c, firstMessage: name } : c)); }, []);

  const selectChat = useCallback(async (sid: string) => {
    setActiveChatId(sid); setStreamText(''); setStreamMeta(null); setIsCorrected(false);
    if (mode !== 'advanced-rag') setMode('advanced-rag');
    try {
      const r = await axios.get(`${API}/rag/chats/${sid}`);
      const { turns } = r.data.data as { sessionId: string; turns: Array<{ id: string; query: string; answer: string; timestamp: string }> };
      const messages: ChatMessage[] = [];
      for (const t of turns) { const ts = t.timestamp ? new Date(t.timestamp).getTime() : Date.now(); messages.push({ role: 'user', content: t.query, timestamp: ts }); messages.push({ role: 'assistant', content: t.answer, timestamp: ts + 1 }); }
      setChats(prev => prev.map(c => c.sessionId === sid ? { ...c, messages } : c));
    } catch {}
  }, [mode]);

  const conversationHistory = (activeChat?.messages ?? []).map(m => ({ role: m.role, content: m.content }));

  const handleAsk = useCallback(async () => {
    if (!question.trim() || isStreaming) return;
    let chatId = activeChatId;
    if (!chatId) {
      const sid = `session_${Date.now()}`;
      setChats(prev => [{ sessionId: sid, firstMessage: question.trim().slice(0, 60), lastActivity: new Date(), turnCount: 0, messages: [] }, ...prev]);
      setActiveChatId(sid); chatId = sid;
    }
    const userMsg: ChatMessage = { role: 'user', content: question.trim(), timestamp: Date.now() };
    setChats(prev => prev.map(c => c.sessionId === chatId ? { ...c, messages: [...(c.messages ?? []), userMsg] } : c));
    const q = question.trim(); setQuestion('');
    abortRef.current?.abort();
    const ctrl = new AbortController(); abortRef.current = ctrl; accumulatedRef.current = '';
    flushSync(() => { setStreamText(''); setStreamMeta(null); setIsCorrected(false); setIsStreaming(true); });
    const body = { question: q, rerankStrategy, includeSources: true, limit, scoreThreshold, temperature, topP, topK, maxTokens, conversationHistory: useConversationMemory ? conversationHistory : undefined, options: { useHybridSearch, useReranking, useQueryTransformation, useContextualCompression, useConversationMemory, sessionId: chatId, useCitationTracking, useKnowledgeGraph } };
    let finalMeta: StreamMeta | null = null; let finalCorrected = false;
    try {
      const res = await fetch(`${API}/rag/documents/ask/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctrl.signal });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader(); const decoder = new TextDecoder(); let tail = '';
      const dispatch = (chunk: StreamChunkEvent) => {
        switch (chunk.event) {
          case 'metadata': setStreamMeta(prev => { const n = { images: [], citations: prev?.citations ?? [], sources: prev?.sources, relevantChunks: prev?.relevantChunks ?? 0, ...chunk.metadata } as StreamMeta; finalMeta = n; return n; }); break;
          case 'sources':  setStreamMeta(prev => { const n = { ...(prev ?? { images: [], citations: [], relevantChunks: 0 }), sources: chunk.sources } as StreamMeta; finalMeta = n; return n; }); break;
          case 'token': accumulatedRef.current += chunk.token; flushSync(() => setStreamText(accumulatedRef.current)); break;
          case 'correction': accumulatedRef.current = chunk.correctedAnswer; finalCorrected = true; flushSync(() => { setStreamText(chunk.correctedAnswer); setIsCorrected(true); }); break;
          case 'citations': setStreamMeta(prev => { const n = { ...(prev ?? { images: [], citations: [], relevantChunks: 0 }), citations: chunk.citations } as StreamMeta; finalMeta = n; return n; }); break;
          case 'done': setStreamMeta(prev => { const n = { images: [], citations: prev?.citations ?? [], sources: prev?.sources, relevantChunks: prev?.relevantChunks ?? 0, ...chunk.metadata } as StreamMeta; finalMeta = n; return n; }); break;
          case 'error': flushSync(() => setStreamText(`⚠ ${chunk.error}`)); break;
        }
      };
      while (true) { const { done, value } = await reader.read(); if (done) break; tail += decoder.decode(value, { stream: true }); const msgs = tail.split('\n\n'); tail = msgs.pop() ?? ''; for (const msg of msgs) { if (!msg.trim()) continue; let d = ''; for (const line of msg.split('\n')) { if (line.startsWith('data: ')) d += line.slice(6); } if (!d.trim()) continue; try { dispatch(JSON.parse(d) as StreamChunkEvent); } catch {} } }
    } catch (e: any) { if (e.name !== 'AbortError') flushSync(() => setStreamText(`⚠ Stream error: ${e.message}`)); }
    finally {
      const fc = accumulatedRef.current;
      if (fc) { setChats(prev => prev.map(c => c.sessionId === chatId ? { ...c, messages: [...(c.messages ?? []), { role: 'assistant', content: fc, meta: finalMeta, isCorrected: finalCorrected, timestamp: Date.now() }] } : c)); setChats(prev => prev.map(c => c.sessionId === chatId ? { ...c, turnCount: (c.turnCount ?? 0) + 1, firstMessage: q.slice(0, 60), lastActivity: new Date() } : c)); }
      setStreamText(''); setStreamMeta(null); setIsStreaming(false);
    }
  }, [question, isStreaming, rerankStrategy, limit, scoreThreshold, temperature, topP, topK, maxTokens, conversationHistory, useHybridSearch, useReranking, useQueryTransformation, useContextualCompression, useConversationMemory, sessionId, useCitationTracking, useKnowledgeGraph, activeChatId]);

  const handleStopStream = () => { abortRef.current?.abort(); setIsStreaming(false); };

  const handleUploadKnowledge = async () => {
    if (!file) return err('Choose a file.');
    const fd = new FormData(); fd.append('file', file); fd.append('chunkingStrategy', chunkingStrategy); fd.append('enableKnowledgeGraph', enableKnowledgeGraph.toString());
    try { setBusy(true); inf('Uploading…'); const r = await axios.post(`${API}/rag/documents/upload`, fd); ok(`${r.data.data?.chunks || 0} chunks · ${chunkingStrategy}`); setFile(null); } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };
  const handleUploadFolder = async () => {
    if (!folderFiles.length) return err('No markdown files selected.');
    const fd = new FormData(); folderFiles.forEach(f => fd.append('files', f)); fd.append('chunkingStrategy', chunkingStrategy); fd.append('enableKnowledgeGraph', enableKnowledgeGraph.toString());
    try { setBusy(true); inf('Uploading folder…'); const r = await axios.post(`${API}/rag/documents/upload-folder`, fd); ok(`${r.data.data.filesProcessed} files → ${r.data.data.totalChunks} chunks`); setFolderFiles([]); } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };
  const handleUploadImages = async () => {
    if (!images.length) return err('Choose images.');
    const fd = new FormData(); images.forEach(img => fd.append('images', img));
    try { setBusy(true); inf('Uploading…'); const r = await axios.post(`${API}/rag/images/upload`, fd); ok(`${r.data.data?.imagesUploaded || 0} uploaded`); setImages([]); if (r.data.data?.generatedImage) setGeneratedImage(`data:image/png;base64,${r.data.data.generatedImage}`); } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };
  const handleImageSearch = async () => {
    if (!question.trim()) return;
    try { setBusy(true); inf('Searching…'); const r = await axios.get(`${API}/rag/images/search`, { params: { query: question, limit: 10 } }); setRetrievedImages(r.data.data); ok(`${r.data.data.length} result(s)`); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  const handleRetrieveAllImages = async () => {
    try { setBusy(true); const r = await axios.get(`${API}/rag/images`); setRetrievedImages(r.data.data); ok(`${r.data.data.length} image(s)`); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  const handleRetrieveAllDocuments = async () => {
    try { setBusy(true); const r = await axios.get(`${API}/rag/documents`); setAllDocuments(r.data.data); ok(`${r.data.data.length} doc(s)`); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  const handleDeleteImage = async (id: string) => {
    try { setBusy(true); await axios.delete(`${API}/rag/images/${id}`); setRetrievedImages(p => p.filter(i => i.id !== id)); ok('Deleted'); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  const handleDeleteDoc = async (id: string) => {
    try { setBusy(true); await axios.delete(`${API}/rag/documents/${id}`); setAllDocuments(p => p.filter(d => d.id !== id)); ok('Deleted'); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  const handleEvaluate = async () => {
    if (!question.trim()) return err('Enter queries');
    try { setBusy(true); inf('Evaluating…'); const queries = question.split('\n').filter(q => q.trim()).map(q => ({ query: q.trim() })); const r = await axios.post(`${API}/rag/documents/evaluate`, { testQueries: queries }); setEvalResults(r.data.data); ok('Done'); } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };
  useEffect(() => { if (mode === 'all-images') handleRetrieveAllImages(); if (mode === 'all-documents') handleRetrieveAllDocuments(); }, [mode]);

  const pp: PipelineProps = { useHybridSearch, setUseHybridSearch, useQueryTransformation, setUseQueryTransformation, useContextualCompression, setUseContextualCompression, useCitationTracking, setUseCitationTracking, useConversationMemory, setUseConversationMemory, useKnowledgeGraph, setUseKnowledgeGraph, includeSources, setIncludeSources, useReranking, setUseReranking, rerankStrategy, setRerankStrategy, limit, setLimit, scoreThreshold, setScoreThreshold, temperature, setTemperature, topP, setTopP, topK, setTopK, maxTokens, setMaxTokens };

  const TABS: { id: UploadMode; label: string }[] = [
    { id: 'advanced-rag', label: 'Chat' }, 
    { id: 'knowledge', label: 'Knowledge' }, 
    //{ id: 'images', label: 'Images' },
    //{ id: 'image-query', label: 'Image Query' }, 
    //{ id: 'all-images', label: 'All Images' }, 
    { id: 'all-documents', label: 'All Docs' }, 
    //{ id: 'evaluation', label: 'Evaluation' },
  ];

  return (
    <div className="h-screen w-screen bg-bg text-text flex flex-col overflow-hidden">
      <ToastContainer toasts={toasts} onRemove={removeToast} />

      {lightboxImg && (
        <div onClick={() => setLightboxImg(null)} className="fixed inset-0 bg-black/90 flex items-center justify-center z-[9999] cursor-zoom-out backdrop-blur-sm">
          <img src={lightboxImg} alt="" className="max-w-[90vw] max-h-[85vh] rounded-xl shadow-[0_0_80px_rgba(0,0,0,.8)]" onClick={e => e.stopPropagation()} />
          <span onClick={() => setLightboxImg(null)} className="absolute top-5 right-6 text-xl text-gray-400 cursor-pointer">✕</span>
        </div>
      )}

      {/* Nav */}
      <nav className="h-[46px] bg-surface border-b border-border flex items-center px-4 flex-shrink-0 overflow-x-auto sticky top-0 z-50">
        <div className="flex items-center gap-2 pr-4 border-r border-border mr-1 flex-shrink-0">
          <div className="w-6 h-6 rounded-[7px] bg-accent flex items-center justify-center font-mono text-[0.58rem] font-bold text-white">R</div>
          <span className="font-sans text-[0.85rem] font-bold text-text">RAG</span>
        </div>
        {TABS.map(tab => (
          <button key={tab.id} onClick={() => setMode(tab.id)}
            className={`bg-transparent border-b-2 cursor-pointer px-3 h-[46px] font-sans text-[0.82rem] whitespace-nowrap transition-all duration-150 flex items-center ${mode === tab.id ? 'text-accent border-b-accent font-semibold' : 'text-muted border-b-transparent hover:text-text'}`}>
            {tab.label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 flex-shrink-0">
          {isStreaming && <Tag color="blue"><Spin size={7} /> live</Tag>}
          <button onClick={() => setDarkMode(d => !d)}
            className="w-8 h-8 rounded-lg border border-border2 bg-surface2 text-muted cursor-pointer flex items-center justify-center text-[0.9rem] transition-all duration-150 hover:border-accent hover:text-accent">
            {darkMode ? '☀' : '☾'}
          </button>
        </div>
      </nav>

      {/* Chat */}
      {mode === 'advanced-rag' && (
        <div className="flex-1 flex overflow-hidden">
          <ChatSidebar chats={chats} activeChatId={activeChatId} onSelect={selectChat} onNew={createNewChat} onDelete={deleteChat} onRename={renameChat} loading={chatsLoading} />
          <div className="flex-1 flex flex-col overflow-hidden relative">
            <div ref={messagesContainerRef} className="flex-1 overflow-y-auto flex flex-col px-8 py-5">
              {hasMessages && <>
                {(activeChat?.messages ?? []).map((msg, i) => <MessageBubble key={i} msg={msg} onLightbox={setLightboxImg} />)}
                {isStreaming && streamText && (
                  <div className="flex flex-col items-start py-1 animate-fade-up">
                    <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center font-mono text-[0.58rem] font-bold text-white flex-shrink-0 mb-1.5">R</div>
                    <div className="max-w-[86%]">
                      <div className="font-sans text-[0.93rem] leading-[1.85] text-text"><MarkdownText text={streamText} /><span className="streaming-cursor" /></div>
                      <div className="flex gap-1 mt-2">
                        {isCorrected && <Tag color="red">hallucination filtered</Tag>}
                        {streamMeta?.queryType && <Tag color={streamMeta.queryType === 'entity' ? 'blue' : streamMeta.queryType === 'wide' ? 'orange' : 'green'}>{streamMeta.queryType}</Tag>}
                        {streamMeta?.confidence !== undefined && <Tag color="green">{(streamMeta.confidence*100).toFixed(0)}% conf</Tag>}
                        {streamMeta?.relevantChunks !== undefined && <Tag>{streamMeta.relevantChunks} chunks</Tag>}
                      </div>
                    </div>
                  </div>
                )}
                {isStreaming && !streamText && (
                  <div className="flex flex-col items-start py-1 animate-fade-up">
                    <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center font-mono text-[0.58rem] font-bold text-white flex-shrink-0 mb-1.5">R</div>
                    <div className="flex gap-1.5 py-2 items-center">
                      {[0,1,2].map(i => <span key={i} className="w-[7px] h-[7px] rounded-full bg-muted inline-block animate-dot-pulse" style={{ animationDelay: `${i*0.2}s` }} />)}
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>}
            </div>
            {showScrollBtn && hasMessages && (
              <button onClick={scrollToBottom} className="absolute bottom-[110px] right-6 w-9 h-9 rounded-full bg-surface border border-border2 text-muted cursor-pointer flex items-center justify-center shadow-[0_4px_12px_rgba(0,0,0,.3)] animate-scroll-bounce hover:border-accent hover:text-accent z-[5] transition-colors">
                <span className="mat-icon">arrow_downward</span>
              </button>
            )}
            <ChatInput question={question} setQuestion={setQuestion} onSend={handleAsk} onStop={handleStopStream} isStreaming={isStreaming} hasMessages={hasMessages} textareaRef={textareaRef} />
          </div>
        </div>
      )}

      {/* Other tabs */}
      {mode !== 'advanced-rag' && (
        <main key={mode} className="animate-tab-fade flex-1 overflow-auto p-5 px-7 max-w-[1140px] w-full mx-auto self-stretch">
          {mode === 'knowledge' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-3.5">
                <Card>
                  <Label>Chunking Strategy</Label>
                  <select value={chunkingStrategy} onChange={e => setChunkingStrategy(e.target.value as ChunkingStrategy)} className="mb-3">
                    <option value="simple">Simple — sentences</option><option value="semantic">Semantic — AI embeddings</option><option value="parent-child">Parent-Child — hierarchical</option>
                  </select>
                  <Toggle checked={enableKnowledgeGraph} onChange={setEnableKnowledgeGraph} label="Extract Knowledge Graph" sub="Build Neo4j entity graph" />
                </Card>
                <Card>
                  <Label>Single File</Label>
                  <input type="file" accept=".docx,.pdf,.txt,.md" onChange={e => { if (e.target.files?.[0]) setFile(e.target.files[0]); }} className="font-mono text-[0.78rem] text-muted mb-2.5 w-full" />
                  {file && <div className="font-mono text-[0.72rem] text-accent mb-2.5">↳ {file.name}</div>}
                  <Btn onClick={handleUploadKnowledge} disabled={busy || !file} accent={!busy && !!file}>{busy ? <span className="flex items-center justify-center gap-2"><Spin/>Uploading…</span> : 'Upload File'}</Btn>
                </Card>
                <Card>
                  <Label>Markdown Folder</Label>
                  <div className="font-mono text-[0.62rem] text-dim mb-1">Chrome / Edge — folder picker</div>
                  <input type="file" {...{ webkitdirectory: '', directory: '' } as any} multiple onChange={e => { if (e.target.files) { const md = Array.from(e.target.files).filter(f => f.name.endsWith('.md')); setFolderFiles(md); md.length ? ok(`${md.length} .md files`) : err('No .md files'); } }} className="font-mono text-[0.78rem] text-muted mb-2.5 w-full" />
                  <div className="font-mono text-[0.62rem] text-dim mb-1">All browsers — multi .md select</div>
                  <input type="file" multiple accept=".md" onChange={e => { if (e.target.files) { const md = Array.from(e.target.files); setFolderFiles(md); ok(`${md.length} file(s)`); } }} className="font-mono text-[0.78rem] text-muted mb-2.5 w-full" />
                  {folderFiles.length > 0 && <div className="font-mono text-[0.7rem] text-accent bg-surface2 rounded-md py-[5px] px-[9px] mb-2.5">{folderFiles.length} files: {folderFiles.slice(0,4).map(f=>f.name).join(', ')}{folderFiles.length>4?` +${folderFiles.length-4}`:''}</div>}
                  <Btn onClick={handleUploadFolder} disabled={busy || !folderFiles.length} accent={!busy && folderFiles.length > 0}>{busy ? <span className="flex items-center justify-center gap-2"><Spin/>Uploading…</span> : `Upload ${folderFiles.length || ''} Files`}</Btn>
                </Card>
              </div>
              <div className="flex flex-col gap-3.5"><PipelinePanel {...pp} /></div>
            </div>
          )}
          {(mode === 'images' || mode === 'image-query') && (
            <div className="grid gap-3.5" style={{ gridTemplateColumns: '300px 1fr' }}>
              <div className="flex flex-col gap-3.5">
                <Card>
                  <Label>Upload Images</Label>
                  <input type="file" accept="image/*" multiple onChange={e => { if (e.target.files) setImages(Array.from(e.target.files)); }} className="font-mono text-[0.78rem] text-muted mb-2 w-full" />
                  <div className="font-mono text-[0.62rem] text-dim mb-2.5">Max 20 · 5 MB each</div>
                  <Btn onClick={handleUploadImages} disabled={busy || !images.length} accent={!busy && images.length > 0}>{busy ? <span className="flex items-center justify-center gap-2"><Spin/>Uploading…</span> : `Upload ${images.length || ''} Image(s)`}</Btn>
                </Card>
                {mode === 'image-query' && (
                  <Card>
                    <Label>Search by Keyword</Label>
                    <input type="text" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') handleImageSearch(); }} placeholder="dog, sunset, city…"
                      className="bg-surface border border-border2 text-text rounded-[10px] px-3.5 py-2.5 text-[0.88rem] outline-none w-full mb-2.5 focus:border-accent focus:shadow-[0_0_0_3px_var(--color-accent-bg)] transition-all duration-150" />
                    <Btn onClick={handleImageSearch} disabled={busy || !question.trim()} accent={!busy && !!question.trim()}>{busy ? <span className="flex items-center justify-center gap-2"><Spin/>Searching…</span> : 'Search'}</Btn>
                  </Card>
                )}
              </div>
              {retrievedImages.length > 0 && mode === 'image-query' && (
                <div>
                  <div className="font-mono text-[0.6rem] font-semibold tracking-[.12em] uppercase text-dim mb-3">{retrievedImages.length} results</div>
                  <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))' }}>
                    {retrievedImages.map(img => (
                      <div key={img.id} onClick={() => setLightboxImg(img.s3Url)} className="rounded-lg overflow-hidden border border-border2 cursor-pointer hover:border-accent hover:scale-[1.02] transition-all duration-150">
                        <img src={img.s3Url} alt="" className="w-full h-[90px] object-cover block" />
                        <div className="p-2">
                          {img.score !== undefined && <div className="font-mono text-[0.65rem] text-accent mb-[3px]">{(img.score*100).toFixed(1)}%</div>}
                          {img.description && <p className="font-mono text-[0.68rem] text-muted leading-snug m-0">{img.description.slice(0,60)}{img.description.length>60?'…':''}</p>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {mode === 'all-images' && (retrievedImages.length > 0
            ? <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(195px,1fr))' }}>
                {retrievedImages.map(img => (
                  <div key={img.id} className="bg-surface border border-border rounded-xl overflow-hidden">
                    <img src={img.s3Url} alt="" className="w-full h-[140px] object-cover block cursor-zoom-in" onClick={() => setLightboxImg(img.s3Url)} />
                    <div className="p-3">
                      {img.description && <p className="font-mono text-[0.72rem] text-muted leading-relaxed mb-2 overflow-hidden" style={{ display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as any }}>{img.description}</p>}
                      {img.keywords && img.keywords.length > 0 && <div className="flex flex-wrap gap-1 mb-2">{img.keywords.slice(0,4).map((kw,i) => <Tag key={i}>{kw}</Tag>)}</div>}
                      <Btn onClick={() => handleDeleteImage(img.id)} disabled={busy} danger>Delete</Btn>
                    </div>
                  </div>
                ))}
              </div>
            : <div className="text-center font-sans text-[0.88rem] text-muted py-20">No images in store</div>
          )}
          {mode === 'all-documents' && (allDocuments.length > 0
            ? <div className="flex flex-col gap-3.5">
                {allDocuments.map(doc => (
                  <Card key={doc.id}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-[0.62rem] text-muted">{doc.id.slice(0,12)}…{doc.createdAt && ` · ${new Date(doc.createdAt).toLocaleDateString()}`}{doc.model && ` · ${doc.model}`}</span>
                      <button onClick={() => handleDeleteDoc(doc.id)} disabled={busy} className="font-mono text-[0.7rem] text-red bg-[color-mix(in_srgb,var(--color-red)_8%,transparent)] border border-[color-mix(in_srgb,var(--color-red)_25%,transparent)] rounded-md py-0.5 px-2.5 cursor-pointer">delete</button>
                    </div>
                    <p className="font-mono text-[0.78rem] text-muted leading-[1.7] m-0">{doc.text.slice(0,300)}{doc.text.length>300?'…':''}</p>
                  </Card>
                ))}
              </div>
            : <div className="text-center font-sans text-[0.88rem] text-muted py-20">No documents in store</div>
          )}
          {mode === 'evaluation' && (
            <div className="max-w-[700px]">
              <Card>
                <Label>Test Queries — one per line</Label>
                <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={10} placeholder={'What is machine learning?\nHow does RAG work?\nExplain transformers…'}
                  className="bg-surface border border-border2 text-text rounded-[10px] px-3.5 py-2.5 text-[0.88rem] outline-none w-full mb-2.5 resize-y leading-[1.7] font-sans focus:border-accent focus:shadow-[0_0_0_3px_var(--color-accent-bg)] transition-all duration-150" />
                <Btn onClick={handleEvaluate} disabled={busy || !question.trim()} accent={!busy && !!question.trim()}>{busy ? <span className="flex items-center justify-center gap-2"><Spin/>Evaluating…</span> : 'Run Evaluation'}</Btn>
              </Card>
              {evalResults && (
                <Card className="mt-3.5">
                  <Label>Results</Label>
                  <div className="grid grid-cols-3 gap-2.5 mb-4">
                    {([['Context Relevance', evalResults.averageMetrics.contextRelevance], ['Faithfulness', evalResults.averageMetrics.answerFaithfulness], ['Answer Relevance', evalResults.averageMetrics.answerRelevance], ['Overall', evalResults.averageMetrics.overall]] as [string, number][]).map(([label, val]) => (
                      <div key={label} className="bg-surface2 border border-border rounded-[10px] p-3.5 text-center">
                        <div className="font-mono text-[0.58rem] text-dim tracking-[.1em] uppercase mb-2">{label}</div>
                        <div className={`font-mono text-[1.4rem] font-bold ${label === 'Overall' ? 'text-accent' : 'text-muted'}`}>{(val*100).toFixed(1)}<span className="text-[0.75rem] font-normal">%</span></div>
                      </div>
                    ))}
                  </div>
                  {evalResults.summary && (
                    <div className="flex gap-5 font-mono text-[0.7rem] text-dim">
                      <span>queries: {evalResults.summary.totalQueries}</span>
                      <span>answered: {evalResults.summary.answeredQueries}</span>
                      {evalResults.summary.avgChunksRetrieved && <span>avg chunks: {evalResults.summary.avgChunksRetrieved.toFixed(1)}</span>}
                    </div>
                  )}
                </Card>
              )}
            </div>
          )}
          {generatedImage && (
            <div className="text-center mt-6">
              <Label>Generated Image</Label>
              <img src={generatedImage} alt="Generated" className="max-w-full rounded-xl border border-border" />
            </div>
          )}
        </main>
      )}
    </div>
  );
};

export default RagDemo;