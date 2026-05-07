import { Button, Card, SectionLabel, Spinner, Toggle } from '../../components/ui/primitives';

type ChunkingStrategy = 'simple' | 'semantic' | 'parent-child';

type Props = {
  chunkingStrategy: ChunkingStrategy;
  setChunkingStrategy: (v: ChunkingStrategy) => void;
  enableKnowledgeGraph: boolean;
  setEnableKnowledgeGraph: (v: boolean) => void;
  file: File | null;
  setFile: (v: File | null) => void;
  folderFiles: File[];
  setFolderFiles: (v: File[]) => void;
  handleUploadKnowledge: () => void;
  handleUploadFolder: () => void;
  busy: boolean;
  ok: (t: string) => void;
  err: (t: string) => void;
};

export const KnowledgeView = ({
  chunkingStrategy,
  setChunkingStrategy,
  enableKnowledgeGraph,
  setEnableKnowledgeGraph,
  file,
  setFile,
  folderFiles,
  setFolderFiles,
  handleUploadKnowledge,
  handleUploadFolder,
  busy,
  ok,
  err,
}: Props) => (
  <div className="mx-auto flex w-full max-w-5xl flex-col gap-5">
    <div>
      <h2 className="text-xl font-semibold text-text">Knowledge Base</h2>
      <p className="text-sm text-muted">Ingest documents into the vector store</p>
    </div>
    <Card>
      <SectionLabel>Chunking Strategy</SectionLabel>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {([
          { val: 'simple', label: 'Simple', sub: 'Split by sentences' },
          { val: 'semantic', label: 'Semantic', sub: 'Embedding-aware chunks' },
          { val: 'parent-child', label: 'Parent Child', sub: 'Hierarchical chunks' },
        ] as const).map(item => (
          <button
            key={item.val}
            type="button"
            onClick={() => setChunkingStrategy(item.val)}
            className={`rounded-xl border p-4 text-left transition ${
              chunkingStrategy === item.val ? 'border-accent bg-accent/10' : 'border-border2 bg-surface2'
            }`}
          >
            <div className="text-sm font-semibold text-text">{item.label}</div>
            <div className="mt-1 text-xs text-muted">{item.sub}</div>
          </button>
        ))}
      </div>
      <div className="mt-2">
        <Toggle
          checked={enableKnowledgeGraph}
          onChange={setEnableKnowledgeGraph}
          label="Extract Knowledge Graph"
          sub="Build Neo4j entity graph"
        />
      </div>
    </Card>

    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <Card className="space-y-3">
        <SectionLabel>Single File</SectionLabel>
        <label className="flex h-28 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-border2 bg-surface2">
          <input
            type="file"
            accept=".docx,.pdf,.txt,.md"
            className="hidden"
            onChange={e => e.target.files?.[0] && setFile(e.target.files[0])}
          />
          <span className="text-sm text-muted">{file ? file.name : 'Drop file or click to browse'}</span>
        </label>
        <Button intent="primary" onClick={handleUploadKnowledge} disabled={busy || !file} className="w-full">
          {busy ? <><Spinner /> Uploading...</> : 'Upload File'}
        </Button>
      </Card>
      <Card className="space-y-3">
        <SectionLabel>Markdown Folder</SectionLabel>
        <label className="flex cursor-pointer items-center justify-between rounded-xl border border-border2 bg-surface2 px-3 py-2.5">
          <span className="text-sm text-muted">Multi-select .md</span>
          <input
            type="file"
            className="hidden"
            multiple
            accept=".md"
            onChange={e => {
              if (e.target.files) {
                const md = Array.from(e.target.files);
                setFolderFiles(md);
                ok(`${md.length} file(s)`);
              }
            }}
          />
        </label>
        {folderFiles.length > 0 ? (
          <p className="text-xs text-muted">
            {folderFiles.length} files selected
          </p>
        ) : null}
        <Button intent="primary" onClick={handleUploadFolder} disabled={busy || !folderFiles.length} className="w-full">
          {busy ? <><Spinner /> Uploading...</> : 'Upload Files'}
        </Button>
        <Button
          intent="ghost"
          className="w-full"
          onClick={() => {
            if (!folderFiles.length) return err('No files to clear');
            setFolderFiles([]);
          }}
        >
          Clear Selected
        </Button>
      </Card>
    </div>
  </div>
);
