import { useEffect, useState } from "react";
import { Sign, listSigns } from "../../api/signs";
import { CreateSignForm } from "./CreateSignForm";

interface SignSelectorProps {
  recentSigns: Sign[];
  selectedSignId: string | null;
  onSelectSign: (signId: string) => void;
  onSignCreated: (signId: string, signName: string) => void;
}

export function SignSelector({
  recentSigns,
  selectedSignId,
  onSelectSign,
  onSignCreated,
}: SignSelectorProps): JSX.Element {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Sign[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  useEffect(() => {
    const search = async () => {
      if (searchQuery.trim().length < 2) {
        setSearchResults([]);
        setSearchError(null);
        return;
      }

      setIsSearching(true);
      setSearchError(null);
      try {
        const response = await listSigns(searchQuery.trim());
        setSearchResults(response.items);
      } catch (error) {
        console.error("Search failed:", error);
        setSearchResults([]);
        setSearchError("Failed to search signs. Please try again.");
      } finally {
        setIsSearching(false);
      }
    };

    const debounce = setTimeout(search, 300);
    return () => clearTimeout(debounce);
  }, [searchQuery]);

  const handleSignCreated = (signId: string, signName: string) => {
    setShowCreateForm(false);
    setSearchQuery("");
    onSignCreated(signId, signName);
  };

  const signsToDisplay = searchQuery.trim().length >= 2 ? searchResults : recentSigns;

  return (
    <div className="space-y-4">
      {/* Search input */}
      <div>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search signs..."
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      {/* Create new sign button */}
      {!showCreateForm && (
        <button
          onClick={() => setShowCreateForm(true)}
          className="w-full px-3 py-2 text-sm font-medium text-primary border border-primary rounded hover:bg-primary/10 transition-colors"
        >
          + Create New Sign
        </button>
      )}

      {/* Create sign form */}
      {showCreateForm && (
        <div className="p-4 bg-slate-800 rounded border border-slate-700">
          <CreateSignForm
            onSignCreated={handleSignCreated}
            onCancel={() => setShowCreateForm(false)}
          />
        </div>
      )}

      {/* Error message */}
      {searchError && (
        <div className="p-3 bg-red-900/20 border border-red-500/30 rounded text-red-300 text-sm">
          {searchError}
        </div>
      )}

      {/* Sign list */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {isSearching ? (
          <div className="text-center text-slate-400 py-4">Searching...</div>
        ) : signsToDisplay.length === 0 ? (
          <div className="text-center text-slate-400 py-4">
            {searchQuery.trim().length >= 2 ? "No signs found" : "No recent signs"}
          </div>
        ) : (
          signsToDisplay.map((sign) => (
            <button
              key={sign.id}
              onClick={() => onSelectSign(sign.id)}
              className={`w-full text-left px-3 py-2 rounded border transition-colors ${
                selectedSignId === sign.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-slate-700 bg-slate-800 text-slate-200 hover:border-slate-600"
              }`}
            >
              <div className="font-medium">{sign.name}</div>
              {sign.description && (
                <div className="text-sm text-slate-400 mt-1">{sign.description}</div>
              )}
            </button>
          ))
        )}
      </div>
    </div>
  );
}
