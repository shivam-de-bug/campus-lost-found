import { useState } from "react";

export default function SearchBar({ onImageSearch, onTextSearch, loading }) {
  const [searchMode, setSearchMode] = useState("image"); // image, text
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchText, setSearchText] = useState("");

  return (
    <div className="bg-white rounded-3xl border border-slate-200/80 shadow-sm p-8 mb-8 animate-slide-up">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-800 m-0">AI Match Search</h3>
        <p className="text-slate-400 text-xs mt-1">Select search mode and query the SigLIP vision-language neural network</p>
      </div>

      {/* Sliding Mode Switcher */}
      <div className="flex bg-slate-100 p-1.5 rounded-2xl mb-6 border border-slate-200/40 w-full max-w-[340px]">
        <button
          onClick={() => setSearchMode("image")}
          className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all duration-300 flex items-center justify-center gap-2 ${
            searchMode === "image"
              ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/10"
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          <i className="fas fa-image text-sm"></i>Search by Photo
        </button>
        <button
          onClick={() => setSearchMode("text")}
          className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all duration-300 flex items-center justify-center gap-2 ${
            searchMode === "text"
              ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/10"
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          <i className="fas fa-keyboard text-sm"></i>Search by Description
        </button>
      </div>

      {/* Image Search Intake */}
      {searchMode === "image" && (
        <div className="space-y-4">
          <label className="block cursor-pointer">
            <div className="border-2 border-dashed border-slate-200 hover:border-indigo-500 hover:bg-slate-50/50 rounded-2xl p-8 text-center transition duration-300 group">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                className="hidden"
              />
              <span className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-500 flex items-center justify-center text-xl mx-auto mb-3 group-hover:scale-110 transition duration-300 shadow-sm">
                <i className="fas fa-cloud-arrow-up"></i>
              </span>
              <p className="text-slate-700 text-sm font-bold">
                {selectedFile ? selectedFile.name : "Choose an image file"}
              </p>
              <p className="text-xs text-slate-400 mt-1">or drag and drop here (JPG, PNG, WebP)</p>
            </div>
          </label>
          <button
            onClick={() => onImageSearch(selectedFile)}
            disabled={!selectedFile || loading}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold py-3.5 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition duration-300 shadow-lg shadow-indigo-600/20 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none flex items-center justify-center gap-2"
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

      {/* Text Search Intake */}
      {searchMode === "text" && (
        <div className="space-y-4">
          <div>
            <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
              Item Details Description
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 text-sm">
                <i className="fas fa-keyboard"></i>
              </span>
              <input
                type="text"
                placeholder="E.g., black leather wallet with IIITD ID card, silver Apple watch..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                className="w-full bg-slate-50/50 border border-slate-200 text-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-400 transition"
              />
            </div>
          </div>
          <button
            onClick={() => onTextSearch(searchText)}
            disabled={!searchText.trim() || loading}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold py-3.5 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition duration-300 shadow-lg shadow-indigo-600/20 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none flex items-center justify-center gap-2"
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
  );
}
