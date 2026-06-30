import { useState, FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

export function UnlockPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [value, setValue] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const token = value.trim();
    if (!token) {
      setError("Token is required.");
      return;
    }
    sessionStorage.setItem("biopb_token", token);
    // Return to the page that sent us here (e.g. /admin); only allow same-site
    // app paths to avoid an open-redirect via the query string.
    const next = searchParams.get("next");
    navigate(next && next.startsWith("/") && !next.startsWith("//") ? next : "/");
  }

  return (
    <main className="unlock-page">
      <div className="unlock-card">
        <h1>BioPB Tensor Viewer</h1>
        <p className="subtitle">
          Enter the access token shown in the launcher terminal to continue.
          <br />
          This token gates the web UI only — it does not affect server data
          access.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="token-input">Access token</label>
            <div className="token-input-row">
              <input
                id="token-input"
                type={showToken ? "text" : "password"}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                autoComplete="off"
                autoFocus
                placeholder="Paste token here"
              />
              <button
                type="button"
                className="reveal-btn"
                onClick={() => setShowToken((v) => !v)}
                aria-label={showToken ? "Hide token" : "Reveal token"}
              >
                {showToken ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {error && <p className="error-msg">{error}</p>}

          <button type="submit" className="submit-btn">
            Unlock
          </button>
        </form>
      </div>

      <style>{`
        .unlock-page {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: #0f1117;
          color: #e2e8f0;
          font-family: system-ui, sans-serif;
        }
        .unlock-card {
          background: #1a1f2e;
          border: 1px solid #2d3748;
          border-radius: 12px;
          padding: 2.5rem 3rem;
          width: 100%;
          max-width: 440px;
        }
        h1 { margin: 0 0 0.5rem; font-size: 1.5rem; }
        .subtitle {
          font-size: 0.875rem;
          color: #94a3b8;
          margin: 0 0 2rem;
          line-height: 1.5;
        }
        .field { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.25rem; }
        label { font-size: 0.875rem; font-weight: 500; }
        .token-input-row { display: flex; gap: 0.5rem; }
        input[type="password"], input[type="text"] {
          flex: 1;
          background: #0f1117;
          border: 1px solid #2d3748;
          border-radius: 6px;
          color: #e2e8f0;
          font-size: 0.9rem;
          padding: 0.625rem 0.875rem;
          outline: none;
        }
        input:focus { border-color: #4f8ef7; }
        .reveal-btn {
          background: #2d3748;
          border: none;
          border-radius: 6px;
          color: #94a3b8;
          cursor: pointer;
          font-size: 0.8rem;
          padding: 0 0.875rem;
          white-space: nowrap;
        }
        .reveal-btn:hover { background: #3d4a5c; color: #e2e8f0; }
        .error-msg { color: #fc8181; font-size: 0.85rem; margin: 0 0 1rem; }
        .submit-btn {
          width: 100%;
          background: #4f8ef7;
          border: none;
          border-radius: 6px;
          color: white;
          cursor: pointer;
          font-size: 0.95rem;
          font-weight: 600;
          padding: 0.75rem;
        }
        .submit-btn:hover { background: #3a7de0; }
      `}</style>
    </main>
  );
}
