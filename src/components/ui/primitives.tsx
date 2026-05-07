import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

export const Badge = ({
  children,
  tone = 'default',
}: {
  children: ReactNode;
  tone?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
}) => (
  <span
    className={cn(
      'inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] font-medium tracking-wide',
      tone === 'default' && 'border-border text-muted',
      tone === 'primary' && 'border-accent/30 bg-accent/10 text-accent',
      tone === 'success' && 'border-emerald-500/35 bg-emerald-500/10 text-emerald-400',
      tone === 'warning' && 'border-amber-500/35 bg-amber-500/10 text-amber-400',
      tone === 'danger' && 'border-red-500/35 bg-red-500/10 text-red-400',
    )}
  >
    {children}
  </span>
);

export const Spinner = ({ size = 12 }: { size?: number }) => (
  <span
    style={{ width: size, height: size }}
    className="inline-block shrink-0 animate-spin rounded-full border-2 border-border border-t-accent"
  />
);

export const Button = ({
  children,
  className,
  intent = 'default',
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  intent?: 'default' | 'primary' | 'danger' | 'ghost';
}) => (
  <button
    className={cn(
      'inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
      intent === 'default' && 'border-border bg-surface text-text hover:border-border2',
      intent === 'primary' && 'border-accent/40 bg-accent/10 text-accent hover:bg-accent hover:text-white',
      intent === 'danger' && 'border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20',
      intent === 'ghost' && 'border-transparent bg-transparent text-muted hover:bg-surface2 hover:text-text',
      className,
    )}
    {...props}
  >
    {children}
  </button>
);

export const Card = ({ children, className }: { children: ReactNode; className?: string }) => (
  <div className={cn('rounded-2xl border border-border bg-surface p-5', className)}>{children}</div>
);

export const SectionLabel = ({ children }: { children: ReactNode }) => (
  <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-dim">{children}</div>
);

export const Toggle = ({
  checked,
  onChange,
  label,
  sub,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  sub?: string;
}) => (
  <button
    type="button"
    onClick={() => onChange(!checked)}
    className="flex w-full items-start gap-3 rounded-xl border border-transparent px-1 py-2 text-left transition hover:border-border"
  >
    <span
      className={cn(
        'relative mt-0.5 h-5 w-9 rounded-full border transition',
        checked ? 'border-accent bg-accent' : 'border-border2 bg-surface2',
      )}
    >
      <span
        className={cn(
          'absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition-all',
          checked ? 'left-4.5' : 'left-0.5 bg-dim',
        )}
      />
    </span>
    <span>
      <span className="text-sm text-text">{label}</span>
      {sub ? <span className="mt-0.5 block text-xs text-muted">{sub}</span> : null}
    </span>
  </button>
);
