import { Badge, Button, Card, Spinner } from '../../components/ui/primitives';

type LinkRecord = { id?: string; sourceFile: string; url: string; title?: string; description?: string };

type LinkQueryResult = { query: string; total: number; links: LinkRecord[]; block?: string } | null;

type LinkMode = 'view' | 'search' | 'query' | 'index';

type Props = {
  links: LinkRecord[];
  linkMode: LinkMode;
  setLinkMode: (v: LinkMode) => void;
  linkSourceFilter: string;
  setLinkSourceFilter: (v: string) => void;
  linkDeleteSource: string;
  setLinkDeleteSource: (v: string) => void;
  linkQuery: string;
  setLinkQuery: (v: string) => void;
  linksLoading: boolean;
  linkQueryResult: LinkQueryResult;
  setLinkQueryResult: (v: LinkQueryResult) => void;
  linkIndexFiles: File[];
  setLinkIndexFiles: (v: File[]) => void;
  handleFetchLinks: () => void;
  handleDeleteLinksBySource: (sourceOverride?: string) => void;
  handleSearchLinks: () => void;
  handleQueryLinks: () => void;
  handleIndexLinks: () => void;
  busy: boolean;
  ok: (t: string) => void;
};

export const LinksView = ({
  links,
  linkMode,
  setLinkMode,
  linkSourceFilter,
  setLinkSourceFilter,
  linkDeleteSource,
  setLinkDeleteSource,
  linkQuery,
  setLinkQuery,
  linksLoading,
  linkQueryResult,
  setLinkQueryResult,
  linkIndexFiles,
  setLinkIndexFiles,
  handleFetchLinks,
  handleDeleteLinksBySource,
  handleSearchLinks,
  handleQueryLinks,
  handleIndexLinks,
  busy,
  ok,
}: Props) => (
  <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
    <div>
      <h2 className="text-xl font-semibold text-text">Knowledge Links</h2>
      <p className="text-sm text-muted">Index, search, query, and manage source links</p>
    </div>
    <div className="flex gap-1 rounded-xl border border-border bg-surface2 p-1">
      {(['view', 'search', 'query', 'index'] as const).map(mode => (
        <button
          key={mode}
          onClick={() => {
            setLinkMode(mode);
            setLinkQueryResult(null);
          }}
          className={`rounded-lg px-3 py-1.5 text-xs uppercase tracking-[0.08em] ${
            linkMode === mode ? 'bg-accent text-white' : 'text-muted'
          }`}
        >
          {mode}
        </button>
      ))}
    </div>
    {linkMode === 'view' ? (
      <Card className="space-y-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={linkSourceFilter}
            onChange={e => setLinkSourceFilter(e.target.value)}
            placeholder="Filter by sourceFile"
            className="flex-1 rounded-xl border border-border2 bg-surface2 px-3 py-2 text-sm text-text outline-none"
          />
          <Button intent="primary" onClick={handleFetchLinks} disabled={linksLoading}>
            {linksLoading ? <><Spinner /> Loading...</> : 'Fetch'}
          </Button>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={linkDeleteSource}
            onChange={e => setLinkDeleteSource(e.target.value)}
            placeholder="sourceFile to delete"
            className="flex-1 rounded-xl border border-border2 bg-surface2 px-3 py-2 text-sm text-text outline-none"
          />
          <Button intent="danger" onClick={() => handleDeleteLinksBySource()} disabled={!linkDeleteSource.trim()}>
            Delete
          </Button>
        </div>
        <div className="space-y-2">
          {links.map((link, i) => (
            <Card key={link.id ?? i} className="p-3">
              <p className="truncate text-xs text-dim">{link.sourceFile}</p>
              <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm text-accent underline">
                {link.url}
              </a>
            </Card>
          ))}
          {!links.length && !linksLoading ? <p className="py-10 text-center text-sm text-muted">No links found.</p> : null}
        </div>
      </Card>
    ) : null}

    {(linkMode === 'search' || linkMode === 'query') && (
      <Card className="space-y-4">
        <div className="flex gap-2">
          <input
            value={linkQuery}
            onChange={e => setLinkQuery(e.target.value)}
            className="flex-1 rounded-xl border border-border2 bg-surface2 px-3 py-2 text-sm text-text outline-none"
          />
          <Button intent="primary" onClick={linkMode === 'search' ? handleSearchLinks : handleQueryLinks}>
            {linkMode === 'search' ? 'Search' : 'Query'}
          </Button>
        </div>
        {linkQueryResult ? (
          <div className="space-y-2">
            <Badge tone="primary">"{linkQueryResult.query}" · {linkQueryResult.total}</Badge>
            {linkQueryResult.links.map((link, i) => (
              <Card key={link.id ?? i} className="p-3">
                <p className="truncate text-xs text-dim">{link.sourceFile}</p>
                <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm text-accent underline">
                  {link.url}
                </a>
              </Card>
            ))}
          </div>
        ) : null}
      </Card>
    )}

    {linkMode === 'index' && (
      <Card className="space-y-3">
        <label className="flex cursor-pointer items-center justify-between rounded-xl border border-border2 bg-surface2 px-3 py-2.5">
          <span className="text-sm text-muted">Select markdown files</span>
          <input
            type="file"
            className="hidden"
            multiple
            accept=".md"
            onChange={e => {
              if (e.target.files) {
                const md = Array.from(e.target.files);
                setLinkIndexFiles(md);
                ok(`${md.length} file(s) selected`);
              }
            }}
          />
        </label>
        <Button intent="primary" disabled={busy || !linkIndexFiles.length} onClick={handleIndexLinks}>
          {busy ? <><Spinner /> Indexing...</> : `Index ${linkIndexFiles.length || ''}`.trim()}
        </Button>
      </Card>
    )}
  </div>
);
