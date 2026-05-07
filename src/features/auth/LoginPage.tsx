import { useState } from 'react';
import { Button, Card } from '../../components/ui/primitives';

type Props = {
  authUser?: string;
  authPass?: string;
  onAuthenticated: () => void;
  onPersistAuth: () => void;
};

export const LoginPage = ({ authUser, authPass, onAuthenticated, onPersistAuth }: Props) => {
  const [user, setUser] = useState('');
  const [pass, setPass] = useState('');
  const [error, setError] = useState('');

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (user.trim() === authUser && pass === authPass) {
      onPersistAuth();
      onAuthenticated();
      return;
    }
    setError('Invalid username or password');
  };

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-bg p-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent text-lg font-bold text-white">
            R
          </div>
          <h1 className="text-xl font-semibold text-text">RAG Workspace</h1>
          <p className="mt-1 text-xs tracking-[0.12em] text-dim">SECURE ACCESS</p>
        </div>
        <Card className="p-6">
          <form onSubmit={submit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-[0.08em] text-muted">Username</label>
              <input
                type="text"
                autoComplete="username"
                autoFocus
                value={user}
                onChange={e => {
                  setUser(e.target.value);
                  setError('');
                }}
                className="w-full rounded-xl border border-border2 bg-surface2 px-3.5 py-2.5 text-sm text-text outline-none transition focus:border-accent"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs uppercase tracking-[0.08em] text-muted">Password</label>
              <input
                type="password"
                autoComplete="current-password"
                value={pass}
                onChange={e => {
                  setPass(e.target.value);
                  setError('');
                }}
                className="w-full rounded-xl border border-border2 bg-surface2 px-3.5 py-2.5 text-sm text-text outline-none transition focus:border-accent"
              />
            </div>
            {error ? <p className="text-sm text-red-400">{error}</p> : null}
            <Button intent="primary" type="submit" className="w-full" disabled={!user.trim() || !pass}>
              Sign In
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
};
