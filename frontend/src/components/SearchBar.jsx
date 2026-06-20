import { useState } from "react";

export default function SearchBar({ onImageSearch, onTextSearch, loading }) {
  const [searchMode, setSearchMode] = useState("image");
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchText, setSearchText] = useState("");

  return (
    <div className="card animate-slide-up" style={{ borderRadius: "var(--radius-2xl)" }}>
      <div className="card-body" style={{ padding: "var(--space-6)" }}>
        <div className="mb-5">
          <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.15rem", margin: 0 }}>
            <i className="fas fa-wand-magic-sparkles mr-2 text-violet"></i>
            AI Match Search
          </h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.75rem", marginTop: 4 }}>
            Search using SigLIP vision-language neural network
          </p>
        </div>

        {/* Mode Switcher */}
        <div className="tab-switcher mb-5" style={{ maxWidth: 340 }}>
          <button
            onClick={() => setSearchMode("image")}
            className={searchMode === "image" ? "active" : ""}
          >
            <i className="fas fa-image mr-1"></i> Search by Photo
          </button>
          <button
            onClick={() => setSearchMode("text")}
            className={searchMode === "text" ? "active" : ""}
          >
            <i className="fas fa-keyboard mr-1"></i> Search by Description
          </button>
        </div>

        {/* Image Search */}
        {searchMode === "image" && (
          <div className="space-y-4">
            <label style={{ display: "block", cursor: "pointer" }}>
              <div className="upload-zone">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  style={{ display: "none" }}
                />
                <div className="upload-zone-icon">
                  <i className="fas fa-cloud-arrow-up"></i>
                </div>
                <p style={{ color: selectedFile ? "var(--primary-light)" : "var(--text-primary)", fontSize: "0.85rem", fontWeight: 700 }}>
                  {selectedFile ? selectedFile.name : "Choose an image file"}
                </p>
                <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4 }}>
                  or drag and drop here (JPG, PNG, WebP)
                </p>
              </div>
            </label>
            <button
              onClick={() => onImageSearch(selectedFile)}
              disabled={!selectedFile || loading}
              className="btn-primary btn-full btn-lg"
            >
              {loading ? (
                <>
                  <i className="fas fa-circle-notch fa-spin"></i>
                  <span>Analyzing visual features...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-search"></i>
                  <span>Run Photo Match</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Text Search */}
        {searchMode === "text" && (
          <div className="space-y-4">
            <div>
              <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
                Item Details Description
              </label>
              <div className="input-icon-wrapper">
                <span className="input-icon"><i className="fas fa-keyboard"></i></span>
                <input
                  type="text"
                  placeholder="E.g., black leather wallet with IIITD ID card, silver Apple watch..."
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                />
              </div>
            </div>
            <button
              onClick={() => onTextSearch(searchText)}
              disabled={!searchText.trim() || loading}
              className="btn-primary btn-full btn-lg"
            >
              {loading ? (
                <>
                  <i className="fas fa-circle-notch fa-spin"></i>
                  <span>Calculating semantic embeddings...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-search"></i>
                  <span>Run Text Match</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
