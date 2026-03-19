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
type StreamMeta = {
  citations: Citation[];
  images: string[];
  confidence?: number;
  relevantChunks: number;
  queryType?: 'entity' | 'factual' | 'wide';
  queryConfidence?: number;
  generationParams?: any;
  sources?: SourceDoc[];
};

type StreamChunkEvent =
  | { event: 'metadata';   metadata: Partial<StreamMeta> & { conversationContext?: boolean } }
  | { event: 'sources';    sources: SourceDoc[] }
  | { event: 'token';      token: string }
  | { event: 'citations';  citations: Citation[] }
  | { event: 'correction'; correctedAnswer: string; reason: 'hallucination' }
  | { event: 'done';       metadata: Partial<StreamMeta> }
  | { event: 'error';      error: string };

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  meta?: StreamMeta | null;
  isCorrected?: boolean;
  timestamp: number;
};

type Chat = {
  sessionId: string;           // primary key — used as chat ID
  firstMessage: string;        // preview label in sidebar
  lastActivity: string | Date;
  turnCount: number;
  messages?: ChatMessage[];    // loaded on demand
};

const API = import.meta.env.VITE_API_URL;

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080808;
    --surface:  #0e0e0e;
    --border:   #181818;
    --border2:  #222;
    --text:     #c8c8c0;
    --muted:    #383830;
    --dim:      #222218;
    --accent:   #c8f560;
    --accent2:  #f5a623;
    --red:      #f25757;
    --blue:     #5b9cf6;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
    --sidebar:  200px;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  ::-webkit-scrollbar { width: 3px; height: 3px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  input[type=range] {
    -webkit-appearance: none; height: 2px;
    background: linear-gradient(90deg, var(--accent) var(--pct, 0%), var(--border2) var(--pct, 0%));
    border-radius: 1px; cursor: pointer; width: 100%;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 10px; height: 10px;
    background: var(--accent); border-radius: 50%; cursor: pointer;
    box-shadow: 0 0 6px rgba(200,245,96,.4);
  }

  @keyframes spin    { to { transform: rotate(360deg); } }
  @keyframes blink   { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes fadeUp  { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  @keyframes slideIn { from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }

  .fade-up  { animation: fadeUp .25s ease both; }
  .slide-in { animation: slideIn .2s ease both; }

  textarea, input[type=text] {
    font-family: var(--mono);
    background: var(--surface);
    border: 1px solid var(--border2);
    color: var(--text);
    border-radius: 6px;
    padding: .6rem .85rem;
    font-size: .82rem;
    outline: none;
    width: 100%;
    resize: vertical;
    transition: border-color .15s;
    line-height: 1.7;
  }
  textarea:focus, input[type=text]:focus { border-color: var(--muted); }

  select {
    font-family: var(--mono);
    background: var(--surface);
    border: 1px solid var(--border2);
    color: var(--text);
    border-radius: 6px;
    padding: .55rem .8rem;
    font-size: .82rem;
    outline: none;
    width: 100%;
    cursor: pointer;
  }

  .tag {
    display: inline-flex; align-items: center; gap: 4px;
    font-family: var(--mono); font-size: .62rem; letter-spacing: .06em;
    padding: 2px 7px; border-radius: 3px;
    border: 1px solid var(--border2); color: var(--muted);
  }
  .tag.green  { border-color: rgba(200,245,96,.25); color: var(--accent);  background: rgba(200,245,96,.05); }
  .tag.orange { border-color: rgba(245,166,35,.25); color: var(--accent2); background: rgba(245,166,35,.05); }
  .tag.red    { border-color: rgba(242,87,87,.25);  color: var(--red);     background: rgba(242,87,87,.05);  }
  .tag.blue   { border-color: rgba(91,156,246,.25); color: var(--blue);    background: rgba(91,156,246,.05); }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .col    { display: flex; flex-direction: column; gap: 14px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }

  .divider { border-top: 1px solid var(--border); margin: 14px 0; padding-top: 14px; }

  .label {
    font-family: var(--mono);
    font-size: .62rem; font-weight: 500;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 8px;
  }

  .streaming-cursor {
    display: inline-block; width: 7px; height: 1em;
    background: var(--accent); margin-left: 2px;
    animation: blink .7s step-end infinite;
    vertical-align: text-bottom; border-radius: 1px;
  }

  .answer-text {
    font-family: var(--mono);
    font-size: .86rem; line-height: 1.85;
    color: #d8d8d0; white-space: pre-wrap;
  }

  .answer-text em { color: #ffffff; font-style: normal; font-weight: 500; }

  .img-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px; margin-top: 10px;
  }
  .img-thumb {
    border-radius: 6px; overflow: hidden;
    border: 1px solid var(--border2);
    cursor: pointer; transition: border-color .15s;
  }
  .img-thumb:hover { border-color: var(--muted); }
  .img-thumb img { width: 100%; height: 90px; object-fit: cover; display: block; }

  .toggle-track {
    flex-shrink: 0; width: 30px; height: 17px; border-radius: 8px;
    border: 1px solid var(--border2); position: relative;
    transition: all .2s; cursor: pointer; margin-top: 2px;
  }
  .toggle-track.on { background: var(--accent); border-color: var(--accent); }
  .toggle-thumb {
    position: absolute; top: 2px; width: 11px; height: 11px;
    border-radius: 50%; transition: left .2s;
  }
  .toggle-thumb.on  { left: 15px; background: #080808; }
  .toggle-thumb.off { left: 2px;  background: var(--border2); }

  /* Sidebar styles */
  .sidebar {
    width: var(--sidebar);
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    height: 100vh;
    position: sticky;
    top: 0;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 16px 14px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .chat-item {
    padding: 8px 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background .15s;
    border-left: 2px solid transparent;
    position: relative;
  }

  .chat-item:hover { background: rgba(255,255,255,.03); }
  .chat-item.active {
    background: rgba(200,245,96,.04);
    border-left-color: var(--accent);
  }

  .chat-item-title {
    font-family: var(--mono);
    font-size: .7rem;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }

  .chat-item.active .chat-item-title { color: var(--text); }

  .chat-item-del {
    opacity: 0;
    font-size: .65rem;
    color: var(--red);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: 3px;
    border: 1px solid rgba(242,87,87,.2);
    background: rgba(242,87,87,.06);
    flex-shrink: 0;
    transition: opacity .15s;
  }
  .chat-item:hover .chat-item-del { opacity: 1; }

  /* Chat messages */
  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 0;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .msg-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    animation: fadeUp .2s ease both;
  }

  .msg-bubble {
    max-width: 80%;
    border-radius: 10px;
    padding: 12px 16px;
  }

  .msg-bubble.user {
    background: rgba(200,245,96,.06);
    border: 1px solid rgba(200,245,96,.12);
    align-self: flex-end;
  }

  .msg-bubble.assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    align-self: flex-start;
  }

  .msg-role {
    font-family: var(--mono);
    font-size: .58rem;
    letter-spacing: .1em;
    color: var(--dim);
    text-transform: uppercase;
    padding: 0 2px;
  }

  .msg-role.user-label { align-self: flex-end; }

  .chat-input-bar {
    padding: 14px 20px;
    border-top: 1px solid var(--border);
    background: var(--bg);
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  .chat-textarea {
    resize: none !important;
    min-height: 44px;
    max-height: 160px;
    overflow-y: auto;
    line-height: 1.6;
    padding: .6rem .85rem;
  }
`;

const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; label: string; sub?: string }> = ({ checked, onChange, label, sub }) => (
  <div onClick={() => onChange(!checked)} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', userSelect: 'none', padding: '4px 0' }}>
    <div className={`toggle-track ${checked ? 'on' : ''}`}>
      <div className={`toggle-thumb ${checked ? 'on' : 'off'}`} />
    </div>
    <div>
      <div style={{ fontSize: '.84rem', color: checked ? 'var(--text)' : 'var(--muted)', transition: 'color .2s' }}>{label}</div>
      {sub && <div style={{ fontSize: '.66rem', color: 'var(--dim)', marginTop: 2, fontFamily: 'var(--mono)' }}>{sub}</div>}
    </div>
  </div>
);

const Btn: React.FC<{ onClick: () => void; disabled?: boolean; danger?: boolean; children: React.ReactNode; accent?: boolean; style?: React.CSSProperties }> = ({ onClick, disabled, danger, children, accent, style }) => (
  <button onClick={onClick} disabled={disabled} style={{
    width: '100%', padding: '.65rem 1.2rem', borderRadius: 7,
    border:      danger ? '1px solid rgba(242,87,87,.2)' : accent ? '1px solid rgba(200,245,96,.3)' : '1px solid var(--border2)',
    background:  danger ? 'rgba(242,87,87,.06)' : accent && !disabled ? 'rgba(200,245,96,.1)' : 'var(--surface)',
    color:       disabled ? 'var(--dim)' : danger ? 'var(--red)' : accent ? 'var(--accent)' : 'var(--muted)',
    cursor:      disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'var(--mono)', fontWeight: 500, fontSize: '.8rem', letterSpacing: '.06em',
    transition: 'all .15s',
    ...style,
  }}>{children}</button>
);

const RangeField: React.FC<{
  label: string; value: number | undefined;
  onChange: (v: number | undefined) => void;
  min: number; max: number; step: number;
  fmt?: (v: number) => string; placeholder?: string;
}> = ({ label, value, onChange, min, max, step, fmt, placeholder = 'auto' }) => {
  const pct = value !== undefined ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>{label}</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', color: value !== undefined ? 'var(--accent)' : 'var(--dim)' }}>
            {value !== undefined ? (fmt ? fmt(value) : value) : placeholder}
          </span>
          {value !== undefined && (
            <span onClick={() => onChange(undefined)} style={{ fontFamily: 'var(--mono)', fontSize: '.6rem', color: 'var(--muted)', cursor: 'pointer', padding: '0 4px', border: '1px solid var(--border2)', borderRadius: 3 }}>✕</span>
          )}
        </div>
      </div>
      <input type="range" min={min} max={max} step={step} value={value ?? min}
        style={{ '--pct': `${pct}%` } as any}
        onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
};

const StatusBar: React.FC<{ msg: { text: string; type: 'ok' | 'err' | 'info' } | null }> = ({ msg }) => {
  if (!msg) return null;
  const cfg = {
    ok:   { bg: 'rgba(200,245,96,.05)', bd: 'rgba(200,245,96,.12)', cl: 'var(--accent)', icon: '✓' },
    err:  { bg: 'rgba(242,87,87,.05)',  bd: 'rgba(242,87,87,.12)',  cl: 'var(--red)',    icon: '✗' },
    info: { bg: 'rgba(91,156,246,.05)', bd: 'rgba(91,156,246,.1)',  cl: 'var(--blue)',   icon: '·' },
  }[msg.type];
  return (
    <div style={{ padding: '.5rem 28px', background: cfg.bg, borderBottom: `1px solid ${cfg.bd}`, fontFamily: 'var(--mono)', fontSize: '.75rem', color: cfg.cl, display: 'flex', gap: 10 }}>
      <span>{cfg.icon}</span><span>{msg.text}</span>
    </div>
  );
};

const Spin: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <span style={{ width: size, height: size, border: `2px solid var(--border2)`, borderTopColor: 'var(--accent)', borderRadius: '50%', display: 'inline-block', animation: 'spin .55s linear infinite', flexShrink: 0 }} />
);

// Utility to generate unique IDs
const uid = () => `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

// Chat Sidebar
const ChatSidebar: React.FC<{
  chats: Chat[];
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  loading?: boolean;
}> = ({ chats, activeChatId, onSelect, onNew, onDelete, loading }) => (
  <div className="sidebar">
    <div className="sidebar-header">
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: '.85rem', fontWeight: 700, letterSpacing: '.12em', color: 'var(--text)' }}>RAG</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: '.52rem', color: 'var(--dim)', letterSpacing: '.18em' }}>SYSTEM</span>
      </div>
      <button onClick={onNew} style={{
        width: '100%', padding: '.5rem', borderRadius: 6,
        border: '1px solid rgba(200,245,96,.2)', background: 'rgba(200,245,96,.06)',
        color: 'var(--accent)', cursor: 'pointer', fontFamily: 'var(--mono)',
        fontSize: '.7rem', letterSpacing: '.06em', fontWeight: 600,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        transition: 'all .15s',
      }}>
        + New Chat
      </button>
    </div>
    <div className="chat-list">
      {loading && (
        <div style={{ padding: '20px 14px', display: 'flex', justifyContent: 'center' }}>
          <Spin size={10} />
        </div>
      )}
      {!loading && chats.length === 0 && (
        <div style={{ padding: '20px 14px', fontFamily: 'var(--mono)', fontSize: '.65rem', color: 'var(--dim)', textAlign: 'center' }}>
          No chats yet
        </div>
      )}
      {chats.map(chat => (
        <div
          key={chat.sessionId}
          className={`chat-item ${activeChatId === chat.sessionId ? 'active' : ''}`}
          onClick={() => onSelect(chat.sessionId)}
        >
          <div className="chat-item-title" title={chat.sessionId}>{chat.firstMessage || chat.sessionId.slice(0, 14)}</div>
          <span
            className="chat-item-del"
            onClick={e => { e.stopPropagation(); onDelete(chat.sessionId); }}
          >✕</span>
        </div>
      ))}
    </div>
    <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)' }}>
      <div style={{ fontFamily: 'var(--mono)', fontSize: '.58rem', color: 'var(--dim)', letterSpacing: '.08em' }}>
        {chats.length} chat{chats.length !== 1 ? 's' : ''}
      </div>
    </div>
  </div>
);

// Message bubble with citations/sources
const MessageBubble: React.FC<{ msg: ChatMessage; onLightbox: (url: string) => void }> = ({ msg, onLightbox }) => {
  const [expandedSrc, setExpandedSrc] = useState(false);
  const isUser = msg.role === 'user';

  return (
    <div className="msg-row">
      <span className={`msg-role ${isUser ? 'user-label' : ''}`}>{isUser ? 'you' : 'rag'}</span>
      <div className={`msg-bubble ${isUser ? 'user' : 'assistant'}`}>
        <div className="answer-text">{msg.content}</div>

        {msg.meta?.images && msg.meta.images.length > 0 && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
            <div className="label">Images ({msg.meta.images.length})</div>
            <div className="img-grid">
              {msg.meta.images.map((url, i) => (
                <div key={i} className="img-thumb" onClick={() => onLightbox(url)}>
                  <img src={url} alt={`img-${i}`} onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                </div>
              ))}
            </div>
          </div>
        )}

        {msg.meta?.citations && msg.meta.citations.length > 0 && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
            <div className="label">Citations ({msg.meta.citations.length})</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {msg.meta.citations.slice(0, 4).map((c, i) => (
                <div key={i} style={{ display: 'flex', gap: 8 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', color: 'var(--accent)', flexShrink: 0 }}>[{i + 1}]</span>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: '.7rem', color: 'var(--muted)', lineHeight: 1.6 }}>
                    {c.text.slice(0, 120)}{c.text.length > 120 ? '…' : ''}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {msg.meta?.sources && msg.meta.sources.length > 0 && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <div className="label" style={{ marginBottom: 0 }}>Sources ({msg.meta.sources.length})</div>
              <span onClick={() => setExpandedSrc(v => !v)} style={{ fontFamily: 'var(--mono)', fontSize: '.6rem', color: 'var(--muted)', cursor: 'pointer', padding: '1px 7px', border: '1px solid var(--border2)', borderRadius: 4 }}>
                {expandedSrc ? 'collapse' : 'expand'}
              </span>
            </div>
            {expandedSrc && msg.meta.sources.map((src, i) => (
              <div key={i} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px', marginBottom: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: '.58rem', color: 'var(--dim)' }}>#{i + 1} · {src.id.slice(0, 8)}…</span>
                  {src.score !== undefined && <span style={{ fontFamily: 'var(--mono)', fontSize: '.58rem', color: 'var(--muted)' }}>{src.score.toFixed(3)}</span>}
                </div>
                <p style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', color: 'var(--muted)', lineHeight: 1.6, margin: 0 }}>
                  {src.text.slice(0, 240)}{src.text.length > 240 ? '…' : ''}
                </p>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: msg.meta ? 10 : 0 }}>
          {msg.isCorrected && <span className="tag red">hallucination filtered</span>}
          {msg.meta?.queryType && <span className={`tag ${msg.meta.queryType === 'entity' ? 'blue' : msg.meta.queryType === 'wide' ? 'orange' : 'green'}`}>{msg.meta.queryType}</span>}
          {msg.meta?.confidence !== undefined && <span className="tag green">{(msg.meta.confidence * 100).toFixed(0)}% conf</span>}
          {msg.meta?.relevantChunks !== undefined && <span className="tag">{msg.meta.relevantChunks} chunks</span>}
        </div>
      </div>
    </div>
  );
};

const RagDemo: React.FC = () => {
  const [mode, setMode] = useState<UploadMode>('advanced-rag');

  // Chat state — synced with backend
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatsLoading, setChatsLoading] = useState(true);

  // Streaming state
  const [streamText, setStreamText] = useState('');
  const [streamMeta, setStreamMeta] = useState<StreamMeta | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isCorrected, setIsCorrected] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const accumulatedRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [question, setQuestion] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [images, setImages] = useState<File[]>([]);
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>('simple');
  const [enableKnowledgeGraph, setEnableKnowledgeGraph] = useState(false);
  const [busy, setBusy] = useState(false);

  // Pipeline state
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
  const [expandedImages] = useState<Record<string, boolean>>({});
  const [evalResults, setEvalResults] = useState<any>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [lightboxImg, setLightboxImg] = useState<string | null>(null);

  const [status, setStatus] = useState<{ text: string; type: 'ok' | 'err' | 'info' } | null>(null);

  const ok  = (t: string) => setStatus({ text: t, type: 'ok' });
  const err = (t: string) => setStatus({ text: t, type: 'err' });
  const inf = (t: string) => setStatus({ text: t, type: 'info' });

  const activeChat = chats.find(c => c.sessionId === activeChatId) ?? null;

  // ── API: fetch all chats on mount ──────────────────────────────────────────
  const fetchChats = useCallback(async () => {
    try {
      setChatsLoading(true);
      const r = await axios.get(`${API}/rag/chats`);
      setChats(r.data.data ?? []);
    } catch {
      // silent — no network connection yet
    } finally {
      setChatsLoading(false);
    }
  }, []);

  useEffect(() => { fetchChats(); }, [fetchChats]);

  // Scroll to bottom when new message arrives
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [streamText, activeChat?.messages?.length]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [question]);

  const createNewChat = useCallback(() => {
    // sessionId is generated on frontend; backend creates the row on first ask
    const sessionId = `session_${Date.now()}`;
    const newChat: Chat = {
      sessionId,
      firstMessage: 'New Chat',
      lastActivity: new Date(),
      turnCount: 0,
      messages: [],
    };
    setChats(prev => [newChat, ...prev]);
    setActiveChatId(sessionId);
    setQuestion('');
    setStreamText('');
    setStreamMeta(null);
    setIsCorrected(false);
    if (mode !== 'advanced-rag') setMode('advanced-rag');
  }, [mode]);

  const deleteChat = useCallback(async (sessionId: string) => {
    try {
      await axios.delete(`${API}/rag/chats/${sessionId}`);
      setChats(prev => {
        const next = prev.filter(c => c.sessionId !== sessionId);
        if (activeChatId === sessionId) {
          setActiveChatId(next.length ? next[0].sessionId : null);
        }
        return next;
      });
    } catch (e: any) {
      err(e.message);
    }
  }, [activeChatId]);

  const selectChat = useCallback(async (sessionId: string) => {
    setActiveChatId(sessionId);
    setStreamText('');
    setStreamMeta(null);
    setIsCorrected(false);
    if (mode !== 'advanced-rag') setMode('advanced-rag');
    // Load full chat turns from backend
    try {
      const r = await axios.get(`${API}/rag/chats/${sessionId}`);
      const { turns } = r.data.data as { sessionId: string; turns: Array<{ id: string; query: string; answer: string; timestamp: string }> };
      // Each turn = one DB row with query + answer → expand to 2 ChatMessage
      const messages: ChatMessage[] = [];
      for (const t of turns) {
        const ts = t.timestamp ? new Date(t.timestamp).getTime() : Date.now();
        messages.push({ role: 'user',      content: t.query,  timestamp: ts });
        messages.push({ role: 'assistant', content: t.answer, timestamp: ts + 1 });
      }
      setChats(prev => prev.map(c => c.sessionId === sessionId ? { ...c, messages } : c));
    } catch {}
  }, [mode]);

  // Build conversation history from active chat
  const conversationHistory = (activeChat?.messages ?? []).map(m => ({ role: m.role, content: m.content }));

  const handleAsk = useCallback(async () => {
    if (!question.trim() || isStreaming) return;

    // Ensure we have an active chat (local sessionId — backend creates row on first answer)
    let chatId = activeChatId;
    if (!chatId) {
      const sessionId = `session_${Date.now()}`;
      const newChat: Chat = {
        sessionId,
        firstMessage: question.trim().slice(0, 60),
        lastActivity: new Date(),
        turnCount: 0,
        messages: [],
      };
      setChats(prev => [newChat, ...prev]);
      setActiveChatId(sessionId);
      chatId = sessionId;
    }

    const userMsg: ChatMessage = { role: 'user', content: question.trim(), timestamp: Date.now() };
    // Optimistic update — messages is optional, guard with ??
    setChats(prev => prev.map(c => c.sessionId === chatId ? { ...c, messages: [...(c.messages ?? []), userMsg] } : c));

    const q = question.trim();
    setQuestion('');

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    accumulatedRef.current = '';
    flushSync(() => {
      setStreamText('');
      setStreamMeta(null);
      setIsCorrected(false);
      setIsStreaming(true);
    });

    const history = conversationHistory;
    const body = {
      question: q,
      rerankStrategy,
      includeSources: true,
      limit,
      scoreThreshold,
      temperature,
      topP,
      topK,
      maxTokens,
      conversationHistory: useConversationMemory ? history : undefined,
      options: {
        useHybridSearch,
        useReranking,
        useQueryTransformation,
        useContextualCompression,
        useConversationMemory,
        sessionId: chatId,
        useCitationTracking,
        useKnowledgeGraph,
      },
    };

    let finalMeta: StreamMeta | null = null;
    let finalCorrected = false;

    try {
      const res = await fetch(`${API}/rag/documents/ask/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal:  ctrl.signal,
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let tail = '';

      const dispatch = (chunk: StreamChunkEvent) => {
        switch (chunk.event) {
          case 'metadata':
            setStreamMeta(prev => {
              const next = {
                images:         [],
                citations:      prev?.citations      ?? [],
                sources:        prev?.sources,
                relevantChunks: prev?.relevantChunks ?? 0,
                ...chunk.metadata,
              } as StreamMeta;
              finalMeta = next;
              return next;
            });
            break;

          case 'sources':
            setStreamMeta(prev => {
              const next = {
                ...(prev ?? { images: [], citations: [], relevantChunks: 0 }),
                sources: chunk.sources,
              } as StreamMeta;
              finalMeta = next;
              return next;
            });
            break;

          case 'token':
            accumulatedRef.current += chunk.token;
            flushSync(() => setStreamText(accumulatedRef.current));
            break;

          case 'correction':
            accumulatedRef.current = chunk.correctedAnswer;
            finalCorrected = true;
            flushSync(() => {
              setStreamText(chunk.correctedAnswer);
              setIsCorrected(true);
            });
            break;

          case 'citations':
            setStreamMeta(prev => {
              const next = {
                ...(prev ?? { images: [], citations: [], relevantChunks: 0 }),
                citations: chunk.citations,
              } as StreamMeta;
              finalMeta = next;
              return next;
            });
            break;

          case 'done':
            setStreamMeta(prev => {
              const next = {
                images:         [],
                citations:      prev?.citations ?? [],
                sources:        prev?.sources,
                relevantChunks: prev?.relevantChunks ?? 0,
                ...chunk.metadata,
              } as StreamMeta;
              finalMeta = next;
              return next;
            });
            break;

          case 'error':
            flushSync(() => setStreamText(`⚠ ${chunk.error}`));
            break;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        tail += decoder.decode(value, { stream: true });
        const messages = tail.split('\n\n');
        tail = messages.pop() ?? '';

        for (const msg of messages) {
          if (!msg.trim()) continue;
          let dataStr = '';
          for (const line of msg.split('\n')) {
            if (line.startsWith('data: ')) dataStr += line.slice(6);
          }
          if (!dataStr.trim()) continue;
          try { dispatch(JSON.parse(dataStr) as StreamChunkEvent); } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        flushSync(() => setStreamText(`⚠ Stream error: ${e.message}`));
      }
    } finally {
      // Save assistant message locally + persist to backend
      const finalContent = accumulatedRef.current;
      if (finalContent) {
        const assistantMsg: ChatMessage = {
          role: 'assistant',
          content: finalContent,
          meta: finalMeta,
          isCorrected: finalCorrected,
          timestamp: Date.now(),
        };
        setChats(prev => prev.map(c => c.sessionId === chatId ? { ...c, messages: [...(c.messages ?? []), assistantMsg] } : c));
        // Backend already persists the turn via ask/stream (ConversationSession.addTurn)
        // Update chat preview in sidebar
        setChats(prev => prev.map(c => c.sessionId === chatId
          ? { ...c, turnCount: (c.turnCount ?? 0) + 1, firstMessage: question.trim().slice(0, 60), lastActivity: new Date() }
          : c
        ));
      }
      setStreamText('');
      setStreamMeta(null);
      setIsStreaming(false);
    }
  }, [question, isStreaming, rerankStrategy, limit, scoreThreshold,
      temperature, topP, topK, maxTokens, conversationHistory, useHybridSearch,
      useReranking, useQueryTransformation, useContextualCompression,
      useConversationMemory, sessionId, useCitationTracking, useKnowledgeGraph,
      activeChatId]);

  const handleStopStream = () => { abortRef.current?.abort(); setIsStreaming(false); };

  // Non-chat mode handlers
  const handleUploadKnowledge = async () => {
    if (!file) return err('Choose a file.');
    const fd = new FormData();
    fd.append('file', file); fd.append('chunkingStrategy', chunkingStrategy);
    fd.append('enableKnowledgeGraph', enableKnowledgeGraph.toString());
    try { setBusy(true); inf('Uploading…');
      const r = await axios.post(`${API}/rag/documents/upload`, fd);
      ok(`${r.data.data?.chunks || 0} chunks · ${chunkingStrategy}`); setFile(null);
    } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };

  const handleUploadFolder = async () => {
    if (!folderFiles.length) return err('No markdown files selected.');
    const fd = new FormData();
    folderFiles.forEach(f => fd.append('files', f));
    fd.append('chunkingStrategy', chunkingStrategy);
    fd.append('enableKnowledgeGraph', enableKnowledgeGraph.toString());
    try { setBusy(true); inf('Uploading folder…');
      const r = await axios.post(`${API}/rag/documents/upload-folder`, fd);
      const { totalChunks, filesProcessed } = r.data.data;
      ok(`${filesProcessed} files → ${totalChunks} chunks`); setFolderFiles([]);
    } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };

  const handleUploadImages = async () => {
    if (!images.length) return err('Choose images.');
    const fd = new FormData(); images.forEach(img => fd.append('images', img));
    try { setBusy(true); inf('Uploading…');
      const r = await axios.post(`${API}/rag/images/upload`, fd);
      ok(`${r.data.data?.imagesUploaded || 0} uploaded`); setImages([]);
      if (r.data.data?.generatedImage) setGeneratedImage(`data:image/png;base64,${r.data.data.generatedImage}`);
    } catch (e: any) { err(e.response?.data?.message || e.message); } finally { setBusy(false); }
  };

  const handleImageSearch = async () => {
    if (!question.trim()) return;
    try { setBusy(true); inf('Searching…');
      const r = await axios.get(`${API}/rag/images/search`, { params: { query: question, limit: 10 } });
      setRetrievedImages(r.data.data); ok(`${r.data.data.length} result(s)`);
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  const handleRetrieveAllImages = async () => {
    try { setBusy(true);
      const r = await axios.get(`${API}/rag/images`); setRetrievedImages(r.data.data); ok(`${r.data.data.length} image(s)`);
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  const handleRetrieveAllDocuments = async () => {
    try { setBusy(true);
      const r = await axios.get(`${API}/rag/documents`); setAllDocuments(r.data.data); ok(`${r.data.data.length} doc(s)`);
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  const handleDeleteImage = async (id: string) => {
    try { setBusy(true); await axios.delete(`${API}/rag/images/${id}`);
      setRetrievedImages(p => p.filter(i => i.id !== id)); ok('Deleted');
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  const handleDeleteDoc = async (id: string) => {
    try { setBusy(true); await axios.delete(`${API}/rag/documents/${id}`);
      setAllDocuments(p => p.filter(d => d.id !== id)); ok('Deleted');
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  const handleEvaluate = async () => {
    if (!question.trim()) return err('Enter queries');
    try { setBusy(true); inf('Evaluating…');
      const queries = question.split('\n').filter(q => q.trim()).map(q => ({ query: q.trim() }));
      const r = await axios.post(`${API}/rag/documents/evaluate`, { testQueries: queries });
      setEvalResults(r.data.data); ok('Done');
    } catch (e: any) { err(e.message); } finally { setBusy(false); }
  };

  useEffect(() => {
    if (mode === 'all-images')     handleRetrieveAllImages();
    if (mode === 'all-documents')  handleRetrieveAllDocuments();
    setStatus(null);
  }, [mode]);

  const renderPipelinePanel = () => (
    <div className="card">
      <div className="label">Pipeline</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 16px' }}>
        <Toggle checked={useHybridSearch}         onChange={setUseHybridSearch}         label="Hybrid Search"   sub="Vector + BM25" />
        <Toggle checked={useQueryTransformation}   onChange={setUseQueryTransformation}   label="Query Expansion" sub="Rephrase + expand" />
        <Toggle checked={useContextualCompression} onChange={setUseContextualCompression} label="Compression"     sub="Extract relevant" />
        <Toggle checked={useCitationTracking}      onChange={setUseCitationTracking}      label="Citations"       sub="Track sources" />
        <Toggle checked={useConversationMemory}    onChange={setUseConversationMemory}    label="Memory"          sub="Session history" />
        <Toggle checked={useKnowledgeGraph}        onChange={setUseKnowledgeGraph}        label="Knowledge Graph" sub="Neo4j enrichment" />
        <Toggle checked={includeSources}           onChange={setIncludeSources}           label="Return Sources"  sub="Attach chunks" />
      </div>

      <div className="divider">
        <div className="label">Re-ranking</div>
        <Toggle checked={useReranking} onChange={setUseReranking} label="Enable Re-ranking" />
        {useReranking && (
          <div style={{ marginTop: 10 }}>
            <div className="label">Strategy</div>
            <select value={rerankStrategy} onChange={e => setRerankStrategy(e.target.value as RerankStrategy)}>
              <option value="none">Hybrid (default)</option>
              <option value="cross_encoder">Cross-encoder (listwise)</option>
              <option value="llm_based">LLM-based</option>
            </select>
          </div>
        )}
      </div>

      <div className="divider">
        <div className="label">Retrieval</div>
        <RangeField label="Chunks"          value={limit}          onChange={setLimit}          min={1}   max={20}   step={1}    placeholder="6" />
        <RangeField label="Score threshold" value={scoreThreshold} onChange={setScoreThreshold} min={0}   max={1}    step={0.05} fmt={v => v.toFixed(2)} placeholder="off" />
      </div>

      <div className="divider">
        <div className="label">LLM Generation</div>
        <RangeField label="Temperature" value={temperature} onChange={setTemperature} min={0}   max={1}    step={0.05} fmt={v => v.toFixed(2)} placeholder="auto" />
        <RangeField label="Top-p"       value={topP}        onChange={setTopP}        min={0}   max={1}    step={0.05} fmt={v => v.toFixed(2)} />
        <RangeField label="Top-k"       value={topK}        onChange={setTopK}        min={1}   max={100}  step={1} />
        <RangeField label="Max tokens"  value={maxTokens}   onChange={setMaxTokens}   min={100} max={8192} step={100} placeholder="auto" />
      </div>
    </div>
  );

  const renderLightbox = () => {
    if (!lightboxImg) return null;
    return (
      <div onClick={() => setLightboxImg(null)} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.88)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 9999, cursor: 'zoom-out', backdropFilter: 'blur(6px)',
      }}>
        <img src={lightboxImg} alt="" style={{ maxWidth: '90vw', maxHeight: '85vh', borderRadius: 10, boxShadow: '0 0 60px rgba(0,0,0,.8)' }} onClick={e => e.stopPropagation()} />
        <span onClick={() => setLightboxImg(null)} style={{ position: 'absolute', top: 20, right: 24, fontFamily: 'var(--mono)', fontSize: '1.2rem', color: 'var(--muted)', cursor: 'pointer' }}>✕</span>
      </div>
    );
  };

  const TABS: { id: UploadMode; label: string }[] = [
    { id: 'advanced-rag',  label: 'Chat'         },
    { id: 'knowledge',     label: 'Knowledge'    },
    { id: 'images',        label: 'Images'       },
    { id: 'image-query',   label: 'Image Query'  },
    { id: 'all-images',    label: 'All Images'   },
    { id: 'all-documents', label: 'All Docs'     },
    { id: 'evaluation',    label: 'Evaluation'   },
  ];

  const renderChatArea = () => (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden', height: '100%' }}>
      {/* Chat sidebar */}
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelect={selectChat}
        onNew={createNewChat}
        onDelete={deleteChat}
        loading={chatsLoading}
      />

      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Messages */}
        <div className="chat-messages" style={{ padding: '20px 24px' }}>
          {!activeChat || (activeChat.messages ?? []).length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 12, padding: '40px 20px' }}>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '1.1rem', color: 'var(--dim)', letterSpacing: '.1em' }}>RAG SYSTEM</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', color: 'var(--dim)', textAlign: 'center', maxWidth: 320, lineHeight: 1.7 }}>
                Ask a question about your knowledge base.<br />Start a new chat or select one from the sidebar.
              </div>
              <button onClick={createNewChat} style={{
                marginTop: 8, padding: '.6rem 1.4rem', borderRadius: 7,
                border: '1px solid rgba(200,245,96,.25)', background: 'rgba(200,245,96,.06)',
                color: 'var(--accent)', cursor: 'pointer', fontFamily: 'var(--mono)',
                fontSize: '.78rem', letterSpacing: '.06em', fontWeight: 600,
              }}>+ Start New Chat</button>
            </div>
          ) : (
            <>
              {(activeChat.messages ?? []).map((msg, i) => (
                <MessageBubble key={i} msg={msg} onLightbox={setLightboxImg} />
              ))}
              {/* Streaming message */}
              {isStreaming && streamText && (
                <div className="msg-row">
                  <span className="msg-role">rag</span>
                  <div className="msg-bubble assistant">
                    <div className="answer-text">
                      {streamText}
                      <span className="streaming-cursor" />
                    </div>
                    <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
                      {isCorrected && <span className="tag red">hallucination filtered</span>}
                      {streamMeta?.queryType && <span className={`tag ${streamMeta.queryType === 'entity' ? 'blue' : streamMeta.queryType === 'wide' ? 'orange' : 'green'}`}>{streamMeta.queryType}</span>}
                      {streamMeta?.confidence !== undefined && <span className="tag green">{(streamMeta.confidence * 100).toFixed(0)}% conf</span>}
                      {streamMeta?.relevantChunks !== undefined && <span className="tag">{streamMeta.relevantChunks} chunks</span>}
                    </div>
                  </div>
                </div>
              )}
              {isStreaming && !streamText && (
                <div className="msg-row">
                  <span className="msg-role">rag</span>
                  <div className="msg-bubble assistant">
                    <Spin size={10} />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input bar */}
        <div className="chat-input-bar">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); handleAsk(); } }}
            placeholder="Ask anything… (⌘↵ to send)"
            rows={1}
            style={{ flex: 1, minHeight: 44, maxHeight: 160 }}
          />
          <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
            {isStreaming ? (
              <button onClick={handleStopStream} style={{
                padding: '.6rem .9rem', borderRadius: 7, border: '1px solid rgba(242,87,87,.25)',
                background: 'rgba(242,87,87,.06)', color: 'var(--red)', cursor: 'pointer',
                fontFamily: 'var(--mono)', fontSize: '.8rem', whiteSpace: 'nowrap',
              }}>■ Stop</button>
            ) : (
              <button onClick={handleAsk} disabled={!question.trim()} style={{
                padding: '.6rem 1.1rem', borderRadius: 7,
                border: question.trim() ? '1px solid rgba(200,245,96,.3)' : '1px solid var(--border2)',
                background: question.trim() ? 'rgba(200,245,96,.1)' : 'var(--surface)',
                color: question.trim() ? 'var(--accent)' : 'var(--dim)',
                cursor: question.trim() ? 'pointer' : 'not-allowed',
                fontFamily: 'var(--mono)', fontSize: '.8rem', fontWeight: 600, whiteSpace: 'nowrap',
                transition: 'all .15s',
              }}>Send ⌘↵</button>
            )}
          </div>
        </div>
      </div>

    </div>
  );

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      {renderLightbox()}

      <div style={{ height: '100vh', width: '100vw', background: 'var(--bg)', color: 'var(--text)', fontFamily: 'var(--sans)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Top nav — only for non-chat tabs */}
        {mode !== 'advanced-rag' && (
          <>
            <header style={{ padding: '1.4rem 28px 0', display: 'flex', alignItems: 'baseline', gap: 12 }}>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '1rem', fontWeight: 700, letterSpacing: '.14em', color: 'var(--text)' }}>RAG</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: '.6rem', color: 'var(--dim)', letterSpacing: '.18em' }}>SYSTEM</div>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                {isStreaming && <span className="tag green"><Spin size={7} /> streaming</span>}
              </div>
            </header>
          </>
        )}

        <nav style={{ display: 'flex', gap: 0, padding: mode !== 'advanced-rag' ? '0 28px' : '0 16px', borderBottom: '1px solid var(--border)', marginTop: mode !== 'advanced-rag' ? 10 : 0, overflowX: 'auto' }}>
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => setMode(tab.id)}
              style={{
                background: 'none', border: 'none',
                borderBottom: mode === tab.id ? '1px solid var(--accent)' : '1px solid transparent',
                color: mode === tab.id ? 'var(--accent)' : 'var(--muted)',
                cursor: 'pointer', padding: '.65rem .9rem',
                fontFamily: 'var(--mono)', fontSize: '.72rem', letterSpacing: '.06em',
                fontWeight: mode === tab.id ? 700 : 400, whiteSpace: 'nowrap',
                transition: 'color .15s',
              }}>
              {tab.label}
            </button>
          ))}
        </nav>

        <StatusBar msg={status} />

        {mode === 'advanced-rag' && (
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            {renderChatArea()}
          </div>
        )}

        {mode !== 'advanced-rag' && (
          <main style={{ flex: 1, padding: '20px 28px', maxWidth: 1120, width: '100%', margin: '0 auto', alignSelf: 'stretch', boxSizing: 'border-box' }}>

            {mode === 'knowledge' && (
              <div className="grid-2">
                <div className="col">
                  <div className="card">
                    <div className="label">Chunking Strategy</div>
                    <select value={chunkingStrategy} onChange={e => setChunkingStrategy(e.target.value as ChunkingStrategy)} style={{ marginBottom: 12 }}>
                      <option value="simple">Simple — sentences</option>
                      <option value="semantic">Semantic — AI embeddings</option>
                      <option value="parent-child">Parent-Child — hierarchical</option>
                    </select>
                    <Toggle checked={enableKnowledgeGraph} onChange={setEnableKnowledgeGraph} label="Extract Knowledge Graph" sub="Build Neo4j entity graph" />
                  </div>

                  <div className="card">
                    <div className="label">Single File</div>
                    <input type="file" accept=".docx,.pdf,.txt,.md" onChange={e => { if (e.target.files?.[0]) setFile(e.target.files[0]); }}
                      style={{ fontFamily: 'var(--mono)', fontSize: '.78rem', color: 'var(--muted)', marginBottom: 10, width: '100%' }} />
                    {file && <div style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', color: 'var(--accent)', marginBottom: 10 }}>↳ {file.name}</div>}
                    <Btn onClick={handleUploadKnowledge} disabled={busy || !file} accent={!busy && !!file}>{busy ? <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Spin />Uploading…</span> : 'Upload File'}</Btn>
                  </div>

                  <div className="card">
                    <div className="label">Markdown Folder</div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', color: 'var(--dim)', marginBottom: 4 }}>Chrome / Edge — folder picker</div>
                    <input type="file" {...{ webkitdirectory: '', directory: '' } as any} multiple onChange={e => { if (e.target.files) { const md = Array.from(e.target.files).filter(f => f.name.endsWith('.md')); setFolderFiles(md); md.length ? ok(`${md.length} .md files selected`) : err('No .md files'); } }}
                      style={{ fontFamily: 'var(--mono)', fontSize: '.78rem', color: 'var(--muted)', marginBottom: 10, width: '100%' }} />
                    <div style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', color: 'var(--dim)', marginBottom: 4 }}>All browsers — multi .md select</div>
                    <input type="file" multiple accept=".md" onChange={e => { if (e.target.files) { const md = Array.from(e.target.files); setFolderFiles(md); ok(`${md.length} file(s)`); } }}
                      style={{ fontFamily: 'var(--mono)', fontSize: '.78rem', color: 'var(--muted)', marginBottom: 10, width: '100%' }} />
                    {folderFiles.length > 0 && <div style={{ fontFamily: 'var(--mono)', fontSize: '.7rem', color: 'var(--accent)', background: 'var(--bg)', borderRadius: 5, padding: '5px 9px', marginBottom: 10 }}>{folderFiles.length} files: {folderFiles.slice(0, 4).map(f => f.name).join(', ')}{folderFiles.length > 4 ? ` +${folderFiles.length - 4}` : ''}</div>}
                    <Btn onClick={handleUploadFolder} disabled={busy || !folderFiles.length} accent={!busy && folderFiles.length > 0}>{busy ? <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Spin />Uploading…</span> : `Upload ${folderFiles.length || ''} Files`}</Btn>
                  </div>
                </div>

                <div className="col">
                  {renderPipelinePanel()}
                </div>
              </div>
            )}

            {(mode === 'images' || mode === 'image-query') && (
              <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 14 }}>
                <div className="col">
                  <div className="card">
                    <div className="label">Upload Images</div>
                    <input type="file" accept="image/*" multiple onChange={e => { if (e.target.files) setImages(Array.from(e.target.files)); }}
                      style={{ fontFamily: 'var(--mono)', fontSize: '.78rem', color: 'var(--muted)', marginBottom: 8, width: '100%' }} />
                    <div style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', color: 'var(--dim)', marginBottom: 10 }}>Max 20 · 5 MB each</div>
                    <Btn onClick={handleUploadImages} disabled={busy || !images.length} accent={!busy && images.length > 0}>{busy ? <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Spin />Uploading…</span> : `Upload ${images.length || ''} Image(s)`}</Btn>
                  </div>
                  {mode === 'image-query' && (
                    <div className="card">
                      <div className="label">Search by Keyword</div>
                      <input type="text" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') handleImageSearch(); }} placeholder="dog, sunset, city…" style={{ marginBottom: 10 }} />
                      <Btn onClick={handleImageSearch} disabled={busy || !question.trim()} accent={!busy && !!question.trim()}>{busy ? <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Spin />Searching…</span> : 'Search'}</Btn>
                    </div>
                  )}
                </div>
                {retrievedImages.length > 0 && mode === 'image-query' && (
                  <div>
                    <div className="label" style={{ marginBottom: 12 }}>{retrievedImages.length} results</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
                      {retrievedImages.map(img => (
                        <div key={img.id} className="img-thumb" onClick={() => setLightboxImg(img.s3Url)}>
                          <img src={img.s3Url} alt={img.description || ''} />
                          <div style={{ padding: '6px 8px' }}>
                            {img.score !== undefined && <div style={{ fontFamily: 'var(--mono)', fontSize: '.65rem', color: 'var(--accent)', marginBottom: 3 }}>{(img.score * 100).toFixed(1)}%</div>}
                            {img.description && <p style={{ fontFamily: 'var(--mono)', fontSize: '.68rem', color: 'var(--muted)', lineHeight: 1.4, margin: 0 }}>{img.description.slice(0, 60)}{img.description.length > 60 ? '…' : ''}</p>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {mode === 'all-images' && (
              retrievedImages.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(195px, 1fr))', gap: 12 }}>
                  {retrievedImages.map(img => (
                    <div key={img.id} className="card" style={{ padding: 0, overflow: 'hidden' }}>
                      <img src={img.s3Url} alt={img.description || ''} style={{ width: '100%', height: 140, objectFit: 'cover', display: 'block', cursor: 'zoom-in' }} onClick={() => setLightboxImg(img.s3Url)} />
                      <div style={{ padding: '10px 12px' }}>
                        {img.description && (
                          <p style={{ fontFamily: 'var(--mono)', fontSize: '.72rem', color: 'var(--muted)', lineHeight: 1.5, margin: '0 0 8px', overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as any }}>
                            {img.description}
                          </p>
                        )}
                        {img.keywords && img.keywords.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                            {img.keywords.slice(0, 4).map((kw, i) => <span key={i} className="tag">{kw}</span>)}
                          </div>
                        )}
                        <Btn onClick={() => handleDeleteImage(img.id)} disabled={busy} danger>Delete</Btn>
                      </div>
                    </div>
                  ))}
                </div>
              ) : <div style={{ textAlign: 'center', fontFamily: 'var(--mono)', fontSize: '.8rem', color: 'var(--dim)', padding: '5rem 0' }}>No images in store</div>
            )}

            {mode === 'all-documents' && (
              allDocuments.length > 0 ? (
                <div className="col">
                  {allDocuments.map(doc => (
                    <div key={doc.id} className="card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: '.62rem', color: 'var(--dim)' }}>
                          {doc.id.slice(0, 12)}…{doc.createdAt && ` · ${new Date(doc.createdAt).toLocaleDateString()}`}
                          {doc.model && ` · ${doc.model}`}
                        </span>
                        <button onClick={() => handleDeleteDoc(doc.id)} disabled={busy}
                          style={{ fontFamily: 'var(--mono)', fontSize: '.7rem', color: 'var(--red)', background: 'rgba(242,87,87,.06)', border: '1px solid rgba(242,87,87,.2)', borderRadius: 5, padding: '2px 10px', cursor: 'pointer' }}>
                          delete
                        </button>
                      </div>
                      <p style={{ fontFamily: 'var(--mono)', fontSize: '.78rem', color: 'var(--muted)', lineHeight: 1.7 }}>
                        {doc.text.slice(0, 300)}{doc.text.length > 300 ? '…' : ''}
                      </p>
                    </div>
                  ))}
                </div>
              ) : <div style={{ textAlign: 'center', fontFamily: 'var(--mono)', fontSize: '.8rem', color: 'var(--dim)', padding: '5rem 0' }}>No documents in store</div>
            )}

            {mode === 'evaluation' && (
              <div style={{ maxWidth: 700 }}>
                <div className="card">
                  <div className="label">Test Queries — one per line</div>
                  <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={10} style={{ marginBottom: 10 }} placeholder={'What is machine learning?\nHow does RAG work?\nExplain transformers…'} />
                  <Btn onClick={handleEvaluate} disabled={busy || !question.trim()} accent={!busy && !!question.trim()}>{busy ? <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Spin />Evaluating…</span> : 'Run Evaluation'}</Btn>
                </div>
                {evalResults && (
                  <div className="card" style={{ marginTop: 14 }}>
                    <div className="label">Results</div>
                    <div className="grid-3" style={{ marginBottom: 16 }}>
                      {([
                        ['Context Relevance', evalResults.averageMetrics.contextRelevance],
                        ['Faithfulness',      evalResults.averageMetrics.answerFaithfulness],
                        ['Answer Relevance',  evalResults.averageMetrics.answerRelevance],
                        ['Overall',          evalResults.averageMetrics.overall],
                      ] as [string, number][]).map(([label, val]) => (
                        <div key={label} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                          <div style={{ fontFamily: 'var(--mono)', fontSize: '.58rem', color: 'var(--dim)', letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 8 }}>{label}</div>
                          <div style={{ fontFamily: 'var(--mono)', fontSize: '1.4rem', fontWeight: 700, color: label === 'Overall' ? 'var(--accent)' : 'var(--muted)' }}>
                            {(val * 100).toFixed(1)}<span style={{ fontSize: '.75rem', fontWeight: 400 }}>%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    {evalResults.summary && (
                      <div style={{ display: 'flex', gap: 20, fontFamily: 'var(--mono)', fontSize: '.7rem', color: 'var(--dim)' }}>
                        <span>queries: {evalResults.summary.totalQueries}</span>
                        <span>answered: {evalResults.summary.answeredQueries}</span>
                        {evalResults.summary.avgChunksRetrieved && <span>avg chunks: {evalResults.summary.avgChunksRetrieved.toFixed(1)}</span>}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {generatedImage && (
              <div style={{ textAlign: 'center', marginTop: 24 }}>
                <div className="label" style={{ marginBottom: 10 }}>Generated Image</div>
                <img src={generatedImage} alt="Generated" style={{ maxWidth: '100%', borderRadius: 10, border: '1px solid var(--border)' }} />
              </div>
            )}
          </main>
        )}
      </div>
    </>
  );
};

export default RagDemo;
