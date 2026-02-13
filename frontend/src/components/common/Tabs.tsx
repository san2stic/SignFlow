import { createContext, useContext, type ReactNode } from "react";

interface TabsContextValue {
  activeTab: string;
  onTabChange: (value: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabsContext(): TabsContextValue {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error("Tab components must be used within TabsWithContext");
  }
  return context;
}

interface TabsWithContextProps {
  activeTab: string;
  onTabChange: (value: string) => void;
  children: ReactNode;
  className?: string;
}

export function TabsWithContext({
  activeTab,
  onTabChange,
  children,
  className = ""
}: TabsWithContextProps): JSX.Element {
  return (
    <TabsContext.Provider value={{ activeTab, onTabChange }}>
      <div className={`flex gap-2 border-b border-slate-700 ${className}`}>
        {children}
      </div>
    </TabsContext.Provider>
  );
}

interface TabProps {
  value: string;
  children: ReactNode;
  badge?: number;
}

export function Tab({ value, children, badge }: TabProps): JSX.Element {
  const { activeTab, onTabChange } = useTabsContext();
  const isActive = activeTab === value;

  return (
    <button
      onClick={() => onTabChange(value)}
      className={`
        relative px-4 py-2 font-medium transition-colors
        ${isActive
          ? "text-primary border-b-2 border-primary"
          : "text-slate-400 hover:text-slate-300"
        }
      `}
    >
      <span className="flex items-center gap-2">
        {children}
        {badge !== undefined && badge > 0 && (
          <span className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1.5 text-xs font-semibold text-white bg-primary rounded-full">
            {badge}
          </span>
        )}
      </span>
    </button>
  );
}
