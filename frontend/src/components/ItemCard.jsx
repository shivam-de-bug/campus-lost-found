import { useState } from "react";
import API from "../api/apiClient";
import ImageLightbox from "./ImageLightbox";

export default function ItemCard({ item, onClaimClick }) {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const imageUrl = item.image_url || (item.filename ? API.getImageUrl(item.filename) : null);

  // Format Date for display
  const displayDate = item.timestamp
    ? new Date(item.timestamp * 1000).toLocaleDateString()
    : item.date
    ? new Date(item.date).toLocaleDateString()
    : "Recently";

  const statusClass = item.status === "claimed"
    ? "badge-success"
    : item.status === "disputed"
    ? "badge-danger"
    : "badge-primary";

  const statusDotClass = item.status === "claimed"
    ? "status-dot-claimed"
    : item.status === "disputed"
    ? "status-dot-disputed"
    : "status-dot-held";

  return (
    <>
      <div className="card animate-slide-up" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", height: "100%" }}>
        <div>
          {/* Image Display */}
          <div className="item-img-container" onClick={() => imageUrl && setLightboxOpen(true)}>
            {imageUrl ? (
              <img
                src={imageUrl}
                alt={item.description || "Found Item"}
                onError={(e) => {
                  e.target.style.display = "none";
                  e.target.nextElementSibling && (e.target.nextElementSibling.style.display = "flex");
                }}
              />
            ) : null}
            <div className="item-no-img" style={imageUrl ? { display: "none" } : {}}>
              <i className="fas fa-image"></i>
              <span>No Image</span>
            </div>

            {/* Category Badge */}
            {item.category && (
              <div style={{ position: "absolute", top: "var(--space-3)", left: "var(--space-3)", zIndex: 1 }}>
                <span className="badge badge-glass">{item.category}</span>
              </div>
            )}

            {/* Status Badge */}
            <div style={{ position: "absolute", top: "var(--space-3)", right: "var(--space-3)", zIndex: 1 }}>
              <span className={`badge ${statusClass}`} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span className={`status-dot ${statusDotClass}`} style={{ width: 6, height: 6 }}></span>
                {item.status || "held"}
              </span>
            </div>
          </div>

          {/* Content */}
          <div className="card-body">
            <h3 style={{
              fontFamily: "'Space Grotesk', sans-serif",
              fontWeight: 700,
              fontSize: "0.95rem",
              color: "var(--text-primary)",
              marginBottom: "var(--space-3)",
              lineHeight: 1.35,
              minHeight: 40,
              transition: "color var(--transition-fast)",
            }} className="line-clamp-2">
              {item.description || "Unidentified Lost Property"}
            </h3>

            <div className="space-y-3">
              <div className="info-row">
                <span className="info-row-icon"><i className="fas fa-map-marker-alt"></i></span>
                <span>Found at: <span className="info-row-value">{item.location}</span></span>
              </div>

              <div className="info-row">
                <span className="info-row-icon"><i className="fas fa-calendar"></i></span>
                <span>Logged: <span className="info-row-value">{displayDate}</span></span>
              </div>

              <div className="info-row">
                <span className="info-row-icon"><i className="fas fa-envelope"></i></span>
                <span className="truncate" style={{ maxWidth: "calc(100% - 30px)" }}>
                  Contact: <span style={{ color: "var(--accent)", fontWeight: 700, fontFamily: "'SF Mono', monospace" }}>{item.contact}</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Handover claims action */}
        {onClaimClick && item.status !== "claimed" && (
          <div style={{ padding: "0 var(--space-5) var(--space-5)" }}>
            <button
              onClick={() => onClaimClick(item)}
              className="btn-primary btn-full"
              style={{ padding: "10px", fontSize: "0.85rem" }}
            >
              <i className="fas fa-handshake"></i>I Claim This Item
            </button>
          </div>
        )}
      </div>

      {/* Image Lightbox */}
      {lightboxOpen && imageUrl && (
        <ImageLightbox
          src={imageUrl}
          alt={item.description}
          onClose={() => setLightboxOpen(false)}
        />
      )}
    </>
  );
}
