import { Button, Card } from '../../components/ui/primitives';

type DocumentData = { id: string; text: string; createdAt?: string; model?: string };

export const DocumentsView = ({
  allDocuments,
  busy,
  handleDeleteDoc,
}: {
  allDocuments: DocumentData[];
  busy: boolean;
  handleDeleteDoc: (id: string) => void;
}) => (
  <div className="mx-auto flex w-full max-w-5xl flex-col gap-4">
    <div>
      <h2 className="text-xl font-semibold text-text">Document Store</h2>
      <p className="text-sm text-muted">{allDocuments.length} chunks indexed</p>
    </div>

    {allDocuments.length > 0 ? (
      <div className="flex flex-col gap-2.5">
        {allDocuments.map((doc, idx) => (
          <Card key={doc.id} className="p-0">
            <div className="flex items-center gap-3 border-b border-border px-4 py-3">
              <span className="rounded-md border border-border2 bg-surface2 px-2 py-1 text-xs text-dim">
                {String(idx + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs text-dim">{doc.id}</p>
              </div>
              <Button intent="danger" onClick={() => handleDeleteDoc(doc.id)} disabled={busy}>
                Delete
              </Button>
            </div>
            <div className="px-4 py-3">
              <p className="line-clamp-3 text-sm leading-6 text-muted">
                {doc.text}
              </p>
            </div>
          </Card>
        ))}
      </div>
    ) : (
      <Card className="py-20 text-center text-sm text-muted">No documents yet. Upload files in Knowledge tab.</Card>
    )}
  </div>
);
