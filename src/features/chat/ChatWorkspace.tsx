import type { ReactNode } from 'react';

export const ChatWorkspace = ({
  sidebar,
  messages,
  input,
}: {
  sidebar: ReactNode;
  messages: ReactNode;
  input: ReactNode;
}) => (
  <div className="flex h-full flex-1 overflow-hidden">
    {sidebar}
    <div className="relative flex flex-1 flex-col overflow-hidden">
      {messages}
      {input}
    </div>
  </div>
);
