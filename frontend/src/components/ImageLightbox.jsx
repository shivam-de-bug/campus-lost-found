import { useEffect } from "react";

export default function ImageLightbox({ src, alt, onClose }) {
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <img
        src={src}
        alt={alt || "Full size preview"}
        className="lightbox-img"
        onClick={(e) => e.stopPropagation()}
      />
      <button className="lightbox-close" onClick={onClose}>
        <i className="fas fa-times"></i>
      </button>
    </div>
  );
}
