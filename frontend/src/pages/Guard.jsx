import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import API from "../api/apiClient";
import { useToast } from "../components/ToastContext";
import Header from "../components/Header";
import ReportForm from "../components/ReportForm";
import { SkeletonGrid } from "../components/Skeleton";
import ImageLightbox from "../components/ImageLightbox";

export default function Guard() {
  const navigate = useNavigate();
  const toast = useToast();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("inventory");
  const [selectedItem, setSelectedItem] = useState(null);
  const [lightboxImg, setLightboxImg] = useState(null);

  const [claimantName, setClaimantName] = useState("");
  const [claimantRoll, setClaimantRoll] = useState("");
  const [verifiedId, setVerifiedId] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const [stats, setStats] = useState({ held: 0, claimed: 0 });

  useEffect(() => { loadInventory(); }, []);

  const loadInventory = async () => {
    setLoading(true);
    try {
      const data = await API.getAllFound();
      const allItems = data.items || [];
      setItems(allItems);
      setStats({
        held: allItems.filter(i => i.status === "held" || !i.status).length,
        claimed: allItems.filter(i => i.status === "claimed").length,
      });
    } catch (error) {
      console.error("Error loading items:", error);
    } finally { setLoading(false); }
  };

  const handleReportFound = async (formData, reportFile) => {
    if (!reportFile || !formData.location || !formData.contact) {
      toast.warning("Please fill all required fields");
      return;
    }
    const form = new FormData();
    form.append("file", reportFile);
    form.append("location", formData.location);
    form.append("contact", formData.contact);
    form.append("description", formData.description);
    form.append("category", formData.category);
    setLoading(true);
    try {
      await API.reportFound(form);
      toast.success("Item logged successfully at gate inventory!");
      setActiveTab("inventory");
      loadInventory();
    } catch (error) {
      toast.error("Error reporting item: " + error.message);
    } finally { setLoading(false); }
  };

  const handleProcessClaim = async (e) => {
    e.preventDefault();
    if (!claimantName || !claimantRoll || !verifiedId) {
      toast.warning("Please complete the verification checks");
      return;
    }
    setActionLoading(true);
    try {
      await API.updateItemStatus(selectedItem.filename, "claimed", claimantRoll, claimantName);
      toast.success("Item released to claimant successfully!");
      setSelectedItem(null);
      setClaimantName("");
      setClaimantRoll("");
      setVerifiedId(false);
      loadInventory();
    } catch (error) {
      toast.error("Error processing claim: " + error.message);
    } finally { setActionLoading(false); }
  };

  const activeHeldItems = items.filter(item => item.status === "held" || !item.status);
  const claimedHistory = items.filter(item => item.status === "claimed");

  return (
    <div className="page animate-fade-in">
      <Header />

      <main className="page-content">
        {/* Page Header + Stats */}
        <div className="flex justify-between items-start flex-wrap gap-4 mb-8 animate-slide-up">
          <div>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "clamp(1.3rem, 4vw, 1.8rem)", margin: 0 }}>
              <i className="fas fa-user-shield mr-2 text-violet"></i>Security Staff Panel
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "var(--space-1)" }}>
              Campus Gate Inventory & Verified Handover
            </p>
          </div>
          <div className="flex gap-3">
            <div className="stat-card" style={{ padding: "var(--space-3) var(--space-5)", textAlign: "center", minWidth: 100 }}>
              <span className="stat-label">Held</span>
              <span className="stat-value" style={{ fontSize: "1.5rem" }}>{stats.held}</span>
            </div>
            <div className="stat-card" style={{ padding: "var(--space-3) var(--space-5)", textAlign: "center", minWidth: 100 }}>
              <span className="stat-label">Claimed</span>
              <span className="stat-value text-success" style={{ fontSize: "1.5rem" }}>{stats.claimed}</span>
            </div>
          </div>
        </div>

        {/* Inventory Tab */}
        {activeTab === "inventory" && (
          <div className="animate-slide-up">
            {loading ? (
              <SkeletonGrid count={4} />
            ) : activeHeldItems.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon"><i className="fas fa-box-open"></i></div>
                <h3>Empty Gate Inventory</h3>
                <p>No items held at any campus gates currently. Newly reported items will appear here.</p>
              </div>
            ) : (
              <div className="grid grid-4">
                {activeHeldItems.map((item, idx) => (
                  <div key={idx} className="card animate-slide-up" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                    <div>
                      <div className="item-img-container" onClick={() => setLightboxImg(API.getImageUrl(item.filename || ""))}>
                        <img
                          src={API.getImageUrl(item.filename || "")}
                          alt={item.description}
                          onError={(e) => { e.target.src = "https://via.placeholder.com/300?text=Gate+Intake"; }}
                        />
                      </div>
                      <div className="card-body">
                        <span className="badge badge-primary mb-2">{item.category || "General"}</span>
                        <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", margin: "var(--space-2) 0", lineHeight: 1.3 }} className="line-clamp-1">
                          {item.description || "Unidentified Item"}
                        </h3>
                        <div className="space-y-3" style={{ marginTop: "var(--space-3)" }}>
                          <div className="info-row">
                            <span className="info-row-icon"><i className="fas fa-map-marker-alt"></i></span>
                            <span className="info-row-value">{item.location}</span>
                          </div>
                          <div className="info-row">
                            <span className="info-row-icon"><i className="fas fa-calendar"></i></span>
                            <span className="info-row-value">{item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Just now"}</span>
                          </div>
                          <div className="info-row">
                            <span className="info-row-icon"><i className="fas fa-envelope"></i></span>
                            <span className="info-row-value">{item.contact}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div style={{ padding: "0 var(--space-5) var(--space-5)" }}>
                      <button onClick={() => setSelectedItem(item)} className="btn-success btn-full" style={{ padding: "10px", fontSize: "0.85rem" }}>
                        <i className="fas fa-hand-holding-hand"></i>Process Claim
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Intake Tab */}
        {activeTab === "intake" && (
          <div className="max-w-2xl mx-auto animate-slide-up">
            <ReportForm
              onSubmit={handleReportFound}
              loading={loading}
              onCancel={() => setActiveTab("inventory")}
            />
          </div>
        )}

        {/* Handover Logs Tab */}
        {activeTab === "logs" && (
          <div className="card animate-slide-up">
            <div className="card-body" style={{ padding: "var(--space-6)" }}>
              <div className="section-header" style={{ marginBottom: "var(--space-5)" }}>
                <i className="fas fa-history"></i>
                <h3>Verified Claim Logs</h3>
              </div>
              {claimedHistory.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon"><i className="fas fa-folder-open"></i></div>
                  <p style={{ fontWeight: 600 }}>No claimed items logged yet.</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Photo</th>
                        <th>Description</th>
                        <th>Claimed By</th>
                        <th>Handed Over By</th>
                        <th>Date</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {claimedHistory.map((item, idx) => (
                        <tr key={idx}>
                          <td>
                            <img src={API.getImageUrl(item.filename)} alt="" style={{ width: 44, height: 44, borderRadius: "var(--radius-md)", objectFit: "cover", border: "1px solid var(--border)" }}
                              onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Img"; }} />
                          </td>
                          <td>
                            <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{item.description || "N/A"}</div>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Found at: {item.location}</div>
                          </td>
                          <td>
                            <div style={{ fontWeight: 600 }}>{item.claimed_by_name || "N/A"}</div>
                            <div style={{ fontSize: "0.7rem", color: "var(--accent)", fontWeight: 700 }}>{item.claimed_by || "N/A"}</div>
                          </td>
                          <td><span className="font-mono text-xs">{item.handed_over_by || "System"}</span></td>
                          <td>{item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Recently"}</td>
                          <td><span className="badge badge-success">Claimed</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Claim Modal */}
      {selectedItem && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">
              <div>
                <h3>Verified Claim Handover</h3>
                <p>Please inspect claimant ID before releasing</p>
              </div>
              <button className="modal-close" onClick={() => setSelectedItem(null)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <form onSubmit={handleProcessClaim} className="space-y-4">
                {/* Item preview */}
                <div className="flex gap-4" style={{ padding: "var(--space-3)", background: "var(--bg-surface)", borderRadius: "var(--radius-xl)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ width: 56, height: 56, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)", border: "1px solid var(--border)", overflow: "hidden", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <img src={API.getImageUrl(selectedItem.filename)} alt="" style={{ width: "100%", height: "100%", objectFit: "contain", padding: 2 }}
                      onError={(e) => { e.target.src = "https://via.placeholder.com/60?text=Item"; }} />
                  </div>
                  <div>
                    <span className="badge badge-primary mb-1">{selectedItem.category}</span>
                    <p style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.9rem" }} className="line-clamp-1">{selectedItem.description}</p>
                    <p style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}><i className="fas fa-map-pin mr-1"></i>Held at: {selectedItem.location}</p>
                  </div>
                </div>

                <div>
                  <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                    Claimant Full Name
                  </label>
                  <input type="text" placeholder="Enter student name" value={claimantName} onChange={(e) => setClaimantName(e.target.value)} required />
                </div>

                <div>
                  <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                    Claimant Roll Number
                  </label>
                  <input type="text" placeholder="E.g., 2023504" value={claimantRoll} onChange={(e) => setClaimantRoll(e.target.value)} required />
                </div>

                <div className="flex gap-3 items-start" style={{ padding: "var(--space-3)", background: "var(--bg-surface)", borderRadius: "var(--radius-lg)", border: "1px solid var(--border-subtle)" }}>
                  <input type="checkbox" id="verify_id" checked={verifiedId} onChange={(e) => setVerifiedId(e.target.checked)} required style={{ marginTop: 2 }} />
                  <label htmlFor="verify_id" style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.5, cursor: "pointer" }}>
                    I confirm that I have verified the claimant's student identity card and details are correct.
                  </label>
                </div>

                <button type="submit" disabled={actionLoading || !verifiedId} className="btn-success btn-full btn-lg" style={{ marginTop: "var(--space-4)" }}>
                  {actionLoading ? (
                    <><i className="fas fa-circle-notch fa-spin"></i><span>Processing Handover...</span></>
                  ) : (
                    <><i className="fas fa-check-circle"></i><span>Release Item</span></>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox */}
      {lightboxImg && <ImageLightbox src={lightboxImg} onClose={() => setLightboxImg(null)} />}

      {/* Bottom Navigation */}
      <nav className="bottom-nav">
        <button onClick={() => setActiveTab("inventory")} className={`nav-item ${activeTab === "inventory" ? "active" : ""}`}>
          <i className="fas fa-boxes-stacked"></i>
          <span>Inventory</span>
        </button>
        <button onClick={() => setActiveTab("intake")} className={`nav-item ${activeTab === "intake" ? "active" : ""}`}>
          <i className="fas fa-plus-circle"></i>
          <span>Intake</span>
        </button>
        <button onClick={() => setActiveTab("logs")} className={`nav-item ${activeTab === "logs" ? "active" : ""}`}>
          <i className="fas fa-clipboard-check"></i>
          <span>Logs</span>
        </button>
        <button onClick={() => navigate("/")} className="nav-item">
          <i className="fas fa-circle-arrow-left"></i>
          <span>Exit</span>
        </button>
      </nav>
    </div>
  );
}
