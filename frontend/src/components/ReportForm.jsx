import { useState } from "react";

export default function ReportForm({ onSubmit, loading, onCancel }) {
  const [formData, setFormData] = useState({
    description: "",
    location: "",
    contact: "",
    category: "",
  });
  const [reportFile, setReportFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const categories = [
    { name: "Electronics", icon: "fa-laptop" },
    { name: "Clothing", icon: "fa-shirt" },
    { name: "Accessories", icon: "fa-glasses" },
    { name: "Documents", icon: "fa-file-alt" },
    { name: "Bags", icon: "fa-bag-shopping" },
    { name: "Footwear", icon: "fa-shoe-prints" },
    { name: "Other", icon: "fa-ellipsis" },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0] || null;
    setReportFile(file);
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await onSubmit(formData, reportFile);
    setFormData({ description: "", location: "", contact: "", category: "" });
    setReportFile(null);
    setPreviewUrl(null);
  };

  return (
    <div className="card animate-slide-up" style={{ borderRadius: "var(--radius-2xl)" }}>
      <div className="card-body" style={{ padding: "var(--space-6)" }}>
        <div className="flex justify-between items-center mb-5">
          <h4 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.1rem", margin: 0 }}>
            <i className="fas fa-plus-circle mr-2 text-violet"></i>Log Found Item
          </h4>
          {onCancel && (
            <button onClick={onCancel} className="btn-icon" style={{ width: 32, height: 32 }}>
              <i className="fas fa-times" style={{ fontSize: "0.75rem" }}></i>
            </button>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Image Upload Zone */}
          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
              Upload Item Photo <span style={{ color: "var(--danger)" }}>*</span>
            </label>
            <label style={{ display: "block", cursor: "pointer" }}>
              <div className="upload-zone" style={{ padding: "var(--space-6) var(--space-4)" }}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setReportFile(e.target.files?.[0] || null)}
                  required
                  style={{ display: "none" }}
                />
                <div className="upload-zone-icon">
                  <i className="fas fa-camera"></i>
                </div>
                {reportFile ? (
                  <p style={{ color: "var(--primary-light)", fontSize: "0.8rem", fontWeight: 700 }}>{reportFile.name}</p>
                ) : (
                  <>
                    <p style={{ color: "var(--text-primary)", fontSize: "0.8rem", fontWeight: 700 }}>Choose an image of the item</p>
                    <p style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: 4 }}>Image matches best when item is centered</p>
                  </>
                )}
              </div>
            </label>
          </div>

          {/* Description */}
          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
              Item Description
            </label>
            <div className="input-icon-wrapper">
              <span className="input-icon" style={{ top: "var(--space-3)", transform: "none" }}><i className="fas fa-pen"></i></span>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleChange}
                placeholder="Describe the item's key features (e.g., color, brand, distinct marks)..."
                rows="3"
                style={{ paddingLeft: "2.5rem" }}
              />
            </div>
          </div>

          {/* Category — Visual Grid */}
          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
              Category <span style={{ color: "var(--danger)" }}>*</span>
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(90px, 1fr))", gap: "var(--space-2)" }}>
              {categories.map((cat) => (
                <button
                  key={cat.name}
                  type="button"
                  onClick={() => setFormData(prev => ({ ...prev, category: cat.name }))}
                  className={formData.category === cat.name ? "role-card active" : "role-card"}
                  style={{ padding: "var(--space-3) var(--space-2)" }}
                >
                  <i className={`fas ${cat.icon}`} style={{ fontSize: "0.9rem", marginBottom: 4 }}></i>
                  <span style={{ fontSize: "0.65rem" }}>{cat.name}</span>
                </button>
              ))}
            </div>
            {/* Hidden input for form validation */}
            <input type="hidden" name="category" value={formData.category} required />
          </div>

          {/* Location */}
          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
              Location Found <span style={{ color: "var(--danger)" }}>*</span>
            </label>
            <div className="input-icon-wrapper">
              <span className="input-icon"><i className="fas fa-map-marker-alt"></i></span>
              <input
                type="text"
                name="location"
                value={formData.location}
                onChange={handleChange}
                placeholder="E.g., Library 2nd floor, Student Center tables..."
                required
              />
            </div>
          </div>

          {/* Contact */}
          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 8 }}>
              Contact Details <span style={{ color: "var(--danger)" }}>*</span>
            </label>
            <div className="input-icon-wrapper">
              <span className="input-icon"><i className="fas fa-envelope"></i></span>
              <input
                type="text"
                name="contact"
                value={formData.contact}
                onChange={handleChange}
                placeholder="E.g., phone number or email address..."
                required
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4" style={{ paddingTop: "var(--space-2)" }}>
            <button
              type="submit"
              disabled={loading}
              className="btn-success flex-1 btn-lg"
            >
              {loading ? (
                <>
                  <i className="fas fa-circle-notch fa-spin"></i>
                  <span>Submitting report...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-circle-check"></i>
                  <span>Submit Report</span>
                </>
              )}
            </button>
            {onCancel && (
              <button type="button" onClick={onCancel} className="btn-secondary flex-1 btn-lg">
                Cancel
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
