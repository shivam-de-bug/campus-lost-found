import API from "../api/apiClient";

export default function ItemCard({ item, onClaimClick }) {
  const imageUrl = item.image_url || (item.filename ? API.getImageUrl(item.filename) : null);
  
  // Format Date for display
  const displayDate = item.timestamp 
    ? new Date(item.timestamp * 1000).toLocaleDateString()
    : (item.date ? new Date(item.date).toLocaleDateString() : "Recently");

  return (
    <div className="bg-white rounded-3xl border border-slate-200/80 hover:border-slate-300 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 overflow-hidden flex flex-col justify-between h-full group">
      <div>
        {/* Image Display */}
        <div className="relative h-48 bg-slate-100 overflow-hidden flex items-center justify-center border-b border-slate-100">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={item.description || "Found Item"}
              className="w-full h-full object-contain p-2 hover:scale-105 transition duration-500"
              onError={(e) => {
                e.target.src = "https://via.placeholder.com/300?text=No+Photo+Available";
              }}
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-300 bg-slate-50">
              <i className="fas fa-image text-4xl mb-2"></i>
              <span className="text-[11px] font-bold uppercase tracking-wider">No Image Provided</span>
            </div>
          )}
          
          {/* Category Badge */}
          {item.category && (
            <div className="absolute top-3 left-3">
              <span className="bg-white/80 backdrop-blur-md text-slate-800 border border-slate-200/50 text-[10px] font-black px-2.5 py-1 rounded-full uppercase tracking-wider shadow-sm">
                {item.category}
              </span>
            </div>
          )}

          {/* Status Badge */}
          <div className="absolute top-3 right-3">
            <span className={`text-[10px] font-black px-2.5 py-1 rounded-full uppercase tracking-wider shadow-sm ${
              item.status === "claimed"
                ? "bg-emerald-600 text-white"
                : item.status === "disputed"
                ? "bg-rose-600 text-white"
                : "bg-indigo-600 text-white"
            }`}>
              {item.status || "held"}
            </span>
          </div>
        </div>

        {/* Content */}
        <div className="p-5">
          <h3 className="font-extrabold text-slate-800 text-base mb-3 leading-snug line-clamp-2 min-h-[44px] group-hover:text-indigo-600 transition">
            {item.description || "Unidentified Lost Property"}
          </h3>

          <div className="space-y-2.5 text-xs text-slate-500 font-medium">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-slate-50 text-slate-400 flex items-center justify-center text-[10px]">
                <i className="fas fa-map-marker-alt"></i>
              </span>
              <span>Found at: <span className="font-bold text-slate-700">{item.location}</span></span>
            </div>

            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-slate-50 text-slate-400 flex items-center justify-center text-[10px]">
                <i className="fas fa-calendar"></i>
              </span>
              <span>Logged: <span className="font-bold text-slate-700">{displayDate}</span></span>
            </div>

            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-slate-50 text-slate-400 flex items-center justify-center text-[10px]">
                <i className="fas fa-envelope"></i>
              </span>
              <span className="truncate">Contact: <span className="font-mono text-indigo-500 font-bold">{item.contact}</span></span>
            </div>
          </div>
        </div>
      </div>

      {/* Handover claims action */}
      {onClaimClick && item.status !== "claimed" && (
        <div className="p-5 pt-0">
          <button
            onClick={() => onClaimClick(item)}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2.5 rounded-xl transition duration-300 shadow-sm flex items-center justify-center gap-2"
          >
            <i className="fas fa-handshake"></i>I Claim This Item
          </button>
        </div>
      )}
    </div>
  );
}
