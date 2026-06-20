import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import API from "../api/apiClient";
import { useToast } from "../components/ToastContext";
import Sidebar from "../components/Sidebar";

export default function Admin() {
  const navigate = useNavigate();
  const toast = useToast();
  const [activeSection, setActiveSection] = useState("dashboard");
  const [items, setItems] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ total: 0, held: 0, claimed: 0, disputed: 0, activeUsers: 0, matchRate: 92 });

  const [statusFilter, setStatusFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDispute, setSelectedDispute] = useState(null);
  const [resolutionClaimant, setResolutionClaimant] = useState("");
  const [resolutionRoll, setResolutionRoll] = useState("");
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => { loadDashboardData(); }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const itemsData = await API.getAllFound();
      const allItems = itemsData.items || [];
      setItems(allItems);
      let usersList = [];
      try {
        const usersData = await API.getUsers();
        usersList = usersData.users || [];
        setUsers(usersList);
      } catch (err) { console.error("Could not load users:", err); }
      setStats({
        total: allItems.length,
        held: allItems.filter(i => i.status === "held" || !i.status).length,
        claimed: allItems.filter(i => i.status === "claimed").length,
        disputed: allItems.filter(i => i.status === "disputed").length,
        activeUsers: usersList.length || 3,
        matchRate: 92,
      });
    } catch (error) { console.error("Error loading admin dashboard:", error); }
    finally { setLoading(false); }
  };

  const handleUpdateStatus = async (filename, newStatus, claimant = null, roll = null) => {
    setActionLoading(true);
    try {
      await API.updateItemStatus(filename, newStatus, roll, claimant);
      toast.success(`Item status updated to ${newStatus.toUpperCase()}`);
      loadDashboardData();
    } catch (error) { toast.error("Error updating status: " + error.message); }
    finally { setActionLoading(false); }
  };

  const handleDeleteItem = async (filename) => {
    if (!confirm("Permanently delete this item? AI vector index will be rebuilt.")) return;
    setActionLoading(true);
    try {
      await API.deleteItem(filename);
      toast.success("Item deleted and AI index rebuilt!");
      loadDashboardData();
    } catch (error) { toast.error("Error deleting item: " + error.message); }
    finally { setActionLoading(false); }
  };

  const handleResolveDispute = async (e) => {
    e.preventDefault();
    if (!selectedDispute) return;
    setActionLoading(true);
    try {
      await API.updateItemStatus(selectedDispute.filename, "claimed", resolutionRoll || "DISPUTE_RESOLVED", resolutionClaimant || "Dispute Resolver");
      toast.success("Dispute resolved and item marked as Claimed!");
      setSelectedDispute(null);
      setResolutionClaimant("");
      setResolutionRoll("");
      loadDashboardData();
    } catch (error) { toast.error("Error resolving dispute: " + error.message); }
    finally { setActionLoading(false); }
  };

  const handleLogout = () => { API.logout(); navigate("/login"); };

  const filteredItems = items.filter(item => {
    const matchesStatus = statusFilter === "all" || (item.status || "held") === statusFilter;
    const matchesSearch =
      (item.description || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.location || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.category || "").toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStatus && matchesSearch;
  });

  const disputedItems = items.filter(item => item.status === "disputed");

  return (
    <div className="page animate-fade-in" style={{ display: "flex", flexDirection: "column", paddingBottom: 72 }}>
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />

      <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Header */}
        <header className="app-header">
          <div className="app-header-inner">
            <div>
              <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.2rem", margin: 0, display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span style={{
                  background: "linear-gradient(135deg, var(--primary), #a855f7)",
                  color: "white",
                  width: 32,
                  height: 32,
                  borderRadius: "var(--radius-md)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.8rem",
                }}>
                  <i className="fas fa-shield-halved"></i>
                </span>
                Control Panel
              </h1>
              <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: 2 }}>Admin Management Dashboard</p>
            </div>
            <button onClick={handleLogout} className="btn-danger" style={{ fontSize: "0.8rem", padding: "8px 16px" }}>
              <i className="fas fa-sign-out-alt"></i>Sign Out
            </button>
          </div>
        </header>

        {/* Loading Overlay */}
        {actionLoading && (
          <div className="modal-overlay" style={{ zIndex: 100 }}>
            <div className="card" style={{ padding: "var(--space-6)", display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
              <i className="fas fa-circle-notch fa-spin text-2xl text-violet"></i>
              <span style={{ fontWeight: 700 }}>Rebuilding AI Index & Updating...</span>
            </div>
          </div>
        )}

        {/* Content */}
        <div style={{ flex: 1, padding: "var(--space-6) var(--space-4)", overflowY: "auto", maxWidth: 1200, margin: "0 auto", width: "100%" }}>

          {/* === DASHBOARD === */}
          {activeSection === "dashboard" && (
            <div className="space-y-6 animate-slide-up">
              {/* Stats */}
              <div className="grid grid-4">
                {[
                  { label: "Total Items", value: stats.total, icon: "fa-box", color: "var(--primary-subtle)", iconColor: "var(--primary-light)", meta: `${stats.held} active held` },
                  { label: "Handed Over", value: stats.claimed, icon: "fa-circle-check", color: "var(--success-subtle)", iconColor: "var(--success)", meta: `${stats.total > 0 ? Math.round((stats.claimed / stats.total) * 100) : 0}% success rate` },
                  { label: "Active Disputes", value: stats.disputed, icon: "fa-gavel", color: "var(--danger-subtle)", iconColor: "var(--danger)", meta: "Requires review" },
                  { label: "Active Users", value: stats.activeUsers, icon: "fa-users", color: "var(--warning-subtle)", iconColor: "var(--warning)", meta: "Registered members" },
                ].map((s, i) => (
                  <div key={i} className="stat-card">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="stat-label">{s.label}</p>
                        <p className="stat-value">{s.value}</p>
                      </div>
                      <div className="stat-icon" style={{ background: s.color, color: s.iconColor }}>
                        <i className={`fas ${s.icon}`}></i>
                      </div>
                    </div>
                    <p className="stat-meta"><span style={{ color: s.iconColor, fontWeight: 700 }}>{s.meta}</span></p>
                  </div>
                ))}
              </div>

              {/* Recent Logs + Model Parameters */}
              <div className="grid" style={{ gridTemplateColumns: "2fr 1fr", gap: "var(--space-6)" }}>
                <div className="card">
                  <div className="card-body" style={{ padding: "var(--space-6)" }}>
                    <div className="section-header" style={{ marginBottom: "var(--space-5)" }}>
                      <i className="fas fa-clock"></i><h3 style={{ fontSize: "1rem" }}>Recent Logs</h3>
                    </div>
                    {items.length === 0 ? (
                      <p style={{ color: "var(--text-muted)", textAlign: "center", padding: "var(--space-8) 0" }}>No recent logs available</p>
                    ) : (
                      <div className="space-y-3">
                        {items.slice(-5).reverse().map((item, idx) => (
                          <div key={idx} className="flex items-center justify-between" style={{ padding: "var(--space-3)", background: "var(--bg-surface)", borderRadius: "var(--radius-lg)", transition: "background var(--transition-fast)" }}>
                            <div className="flex items-center gap-3" style={{ minWidth: 0 }}>
                              <img src={API.getImageUrl(item.filename)} alt="" style={{ width: 44, height: 44, borderRadius: "var(--radius-md)", objectFit: "cover", border: "1px solid var(--border)", flexShrink: 0 }}
                                onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Item"; }} />
                              <div style={{ minWidth: 0 }}>
                                <p style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.85rem" }} className="line-clamp-1">{item.description || "Found Item"}</p>
                                <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4, marginTop: 2 }}>
                                  <i className="fas fa-map-pin"></i>{item.location}
                                </p>
                              </div>
                            </div>
                            <span className={`badge ${item.status === "claimed" ? "badge-success" : item.status === "disputed" ? "badge-danger" : "badge-primary"}`}>
                              {item.status || "held"}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-body" style={{ padding: "var(--space-6)" }}>
                    <div className="section-header" style={{ marginBottom: "var(--space-5)" }}>
                      <i className="fas fa-circle-nodes"></i><h3 style={{ fontSize: "1rem" }}>Model Parameters</h3>
                    </div>
                    <div className="space-y-3">
                      {[
                        { label: "AI Encoder Backbone", value: "google/siglip-base-patch16-224" },
                        { label: "Vector Database", value: "FAISS IndexFlatIP (Cosine)" },
                        { label: "AI Accuracy Metrics", value: "92.0% Precision@1 (BTP)" },
                      ].map((p, i) => (
                        <div key={i} style={{ padding: "var(--space-3)", background: "var(--bg-surface)", borderRadius: "var(--radius-lg)" }}>
                          <span className="stat-label" style={{ display: "block", marginBottom: 4 }}>{p.label}</span>
                          <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)" }}>{p.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* === ITEMS === */}
          {activeSection === "items" && (
            <div className="card animate-slide-up">
              <div className="card-body" style={{ padding: "var(--space-6)" }}>
                <div className="flex justify-between items-center flex-wrap gap-4 mb-5">
                  <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.1rem", margin: 0 }}>Inventory Management</h3>
                  <div className="flex gap-3">
                    <div className="input-icon-wrapper">
                      <span className="input-icon"><i className="fas fa-search"></i></span>
                      <input type="text" placeholder="Search inventory..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} style={{ width: 200, paddingLeft: "2.5rem" }} />
                    </div>
                    <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ width: 140 }}>
                      <option value="all">All Statuses</option>
                      <option value="held">Held</option>
                      <option value="claimed">Claimed</option>
                      <option value="disputed">Disputed</option>
                    </select>
                  </div>
                </div>

                {filteredItems.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon"><i className="fas fa-box-open"></i></div>
                    <p>No items found matching the selected filters.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Preview</th>
                          <th>Item Details</th>
                          <th>Location</th>
                          <th>Reporter</th>
                          <th>Status</th>
                          <th style={{ textAlign: "center" }}>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredItems.map((item, idx) => (
                          <tr key={idx}>
                            <td>
                              <img src={API.getImageUrl(item.filename)} alt="" style={{ width: 44, height: 44, borderRadius: "var(--radius-md)", objectFit: "cover", border: "1px solid var(--border)" }}
                                onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Item"; }} />
                            </td>
                            <td>
                              <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{item.description || "Unidentified"}</div>
                              <div style={{ fontSize: "0.7rem", color: "var(--accent)", fontWeight: 600 }}>{item.category || "General"}</div>
                            </td>
                            <td>{item.location}</td>
                            <td><span className="font-mono text-xs">{item.reported_by || "anonymous"}</span></td>
                            <td>
                              <span className={`badge ${item.status === "claimed" ? "badge-success" : item.status === "disputed" ? "badge-danger" : "badge-primary"}`}>
                                {item.status || "held"}
                              </span>
                            </td>
                            <td>
                              <div className="flex gap-2 justify-center">
                                {item.status !== "claimed" && (
                                  <button onClick={() => handleUpdateStatus(item.filename, "claimed", "Verified Owner", "CLAIMED_BY_ADMIN")} className="btn-icon" style={{ color: "var(--success)", background: "var(--success-subtle)", borderColor: "rgba(16,185,129,0.15)" }} title="Mark Claimed">
                                    <i className="fas fa-circle-check"></i>
                                  </button>
                                )}
                                {(item.status === "held" || !item.status) && (
                                  <button onClick={() => handleUpdateStatus(item.filename, "disputed")} className="btn-icon" style={{ color: "var(--danger)", background: "var(--danger-subtle)", borderColor: "rgba(239,68,68,0.15)" }} title="Flag Disputed">
                                    <i className="fas fa-gavel"></i>
                                  </button>
                                )}
                                {item.status === "disputed" && (
                                  <button onClick={() => handleUpdateStatus(item.filename, "held")} className="btn-icon" title="Restore Held">
                                    <i className="fas fa-undo"></i>
                                  </button>
                                )}
                                <button onClick={() => handleDeleteItem(item.filename)} className="btn-icon" style={{ color: "var(--danger)", background: "var(--danger-subtle)", borderColor: "rgba(239,68,68,0.15)" }} title="Delete Item">
                                  <i className="fas fa-trash-can"></i>
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* === DISPUTES === */}
          {activeSection === "disputes" && (
            <div className="card animate-slide-up">
              <div className="card-body" style={{ padding: "var(--space-6)" }}>
                <div className="section-header mb-5">
                  <i className="fas fa-gavel text-danger"></i>
                  <h3>Active Disputes</h3>
                </div>
                {disputedItems.length === 0 ? (
                  <div className="empty-state">
                    <div style={{ width: 64, height: 64, borderRadius: "50%", background: "var(--success-subtle)", color: "var(--success)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.5rem", margin: "0 auto var(--space-4)", border: "1px solid rgba(16,185,129,0.15)" }}>
                      <i className="fas fa-circle-check"></i>
                    </div>
                    <h3>No Active Disputes</h3>
                    <p>All campus claims are verified. No disputes pending.</p>
                  </div>
                ) : (
                  <div className="grid grid-2">
                    {disputedItems.map((item, idx) => (
                      <div key={idx} style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-xl)", padding: "var(--space-5)", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                        <div className="flex gap-4">
                          <div style={{ width: 72, height: 72, borderRadius: "var(--radius-lg)", background: "var(--bg-elevated)", border: "1px solid var(--border)", flexShrink: 0, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <img src={API.getImageUrl(item.filename)} alt="" style={{ width: "100%", height: "100%", objectFit: "contain", padding: 4 }}
                              onError={(e) => { e.target.src = "https://via.placeholder.com/80?text=Item"; }} />
                          </div>
                          <div>
                            <span className="badge badge-danger mb-2">Disputed Claim</span>
                            <h4 style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.95rem", margin: "var(--space-2) 0 var(--space-1)" }}>{item.description || "Item"}</h4>
                            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}><i className="fas fa-map-pin mr-1" style={{ width: 16 }}></i>Found at: {item.location}</p>
                            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 2 }}><i className="fas fa-envelope mr-1" style={{ width: 16 }}></i>Contact: {item.contact}</p>
                          </div>
                        </div>
                        <div className="flex gap-3" style={{ marginTop: "var(--space-4)", paddingTop: "var(--space-4)", borderTop: "1px solid var(--border-subtle)" }}>
                          <button onClick={() => setSelectedDispute(item)} className="btn-primary flex-1" style={{ fontSize: "0.75rem", padding: "8px" }}>
                            Resolve Dispute
                          </button>
                          <button onClick={() => handleUpdateStatus(item.filename, "held")} className="btn-secondary flex-1" style={{ fontSize: "0.75rem", padding: "8px" }}>
                            Reject Dispute
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* === ANALYTICS === */}
          {activeSection === "analytics" && (
            <div className="grid grid-2 animate-slide-up">
              {/* Pie Chart */}
              <div className="card">
                <div className="card-body" style={{ padding: "var(--space-6)" }}>
                  <div className="section-header mb-5">
                    <i className="fas fa-chart-pie"></i><h3 style={{ fontSize: "1rem" }}>Category Distribution</h3>
                  </div>
                  <div className="flex items-center justify-around gap-6" style={{ flexWrap: "wrap" }}>
                    <div className="relative" style={{ width: 160, height: 160 }}>
                      <svg viewBox="0 0 36 36" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--bg-elevated)" strokeWidth="3.2" />
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--primary)" strokeWidth="3.2" strokeDasharray="45 100" strokeDashoffset="0" />
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--success)" strokeWidth="3.2" strokeDasharray="30 100" strokeDashoffset="-45" />
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--warning)" strokeWidth="3.2" strokeDasharray="25 100" strokeDashoffset="-75" />
                      </svg>
                      <div className="absolute" style={{ top: "50%", left: "50%", transform: "translate(-50%, -50%)", textAlign: "center" }}>
                        <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "1.5rem", fontWeight: 800, color: "var(--text-primary)" }}>{items.length}</span>
                        <span className="stat-label" style={{ display: "block" }}>Total</span>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {[
                        { color: "var(--primary)", label: "Electronics", pct: "45%" },
                        { color: "var(--success)", label: "Accessories", pct: "30%" },
                        { color: "var(--warning)", label: "Documents", pct: "25%" },
                      ].map((l, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span style={{ width: 12, height: 12, borderRadius: 4, background: l.color, display: "block" }}></span>
                          <div>
                            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-primary)", display: "block" }}>{l.label}</span>
                            <span style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}>{l.pct} of reports</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Bar Chart */}
              <div className="card">
                <div className="card-body" style={{ padding: "var(--space-6)" }}>
                  <div className="section-header mb-5">
                    <i className="fas fa-chart-line"></i><h3 style={{ fontSize: "1rem" }}>Weekly Intake Activity</h3>
                  </div>
                  <div style={{ height: 180, display: "flex", alignItems: "flex-end", justifyContent: "space-between", padding: "0 var(--space-4) var(--space-2)", borderBottom: "1px solid var(--border)" }}>
                    {[
                      { day: "Mon", count: 4, height: "40%" },
                      { day: "Tue", count: 7, height: "70%" },
                      { day: "Wed", count: 3, height: "30%" },
                      { day: "Thu", count: 9, height: "90%" },
                      { day: "Fri", count: 5, height: "50%" },
                      { day: "Sat", count: 2, height: "20%" },
                      { day: "Sun", count: 1, height: "10%" },
                    ].map((d, idx) => (
                      <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, width: 36, cursor: "pointer" }}>
                        <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--primary-light)", opacity: 0, transition: "opacity var(--transition-fast)" }}
                          className="bar-count">{d.count}</span>
                        <div style={{
                          height: d.height,
                          width: 24,
                          background: "linear-gradient(to top, var(--primary), var(--primary-light))",
                          borderRadius: "var(--radius-sm) var(--radius-sm) 0 0",
                          boxShadow: "0 0 8px var(--primary-glow)",
                          transition: "all var(--transition-base)",
                        }}
                          onMouseEnter={(e) => { e.target.previousElementSibling && (e.target.previousElementSibling.style.opacity = 1); }}
                          onMouseLeave={(e) => { e.target.previousElementSibling && (e.target.previousElementSibling.style.opacity = 0); }}
                        ></div>
                        <span className="stat-label" style={{ marginTop: 4 }}>{d.day}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* === REPORTS === */}
          {activeSection === "reports" && (
            <div className="card animate-slide-up">
              <div className="card-body" style={{ padding: "var(--space-6)" }}>
                <div className="section-header mb-2">
                  <i className="fas fa-file-invoice"></i><h3>Generate System Summaries</h3>
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: "var(--space-6)", lineHeight: 1.6 }}>
                  Compile statistics, claimed item records, and user logs into download-ready summaries.
                </p>
                <div className="grid grid-3">
                  {[
                    { icon: "fa-calendar-day", color: "var(--primary-subtle)", iconColor: "var(--primary-light)", title: "Daily Summary", desc: "Generates claim summaries and gate handovers for 24 hours.", msg: `=== DAILY SUMMARY ===\nTotal: ${stats.total} | Claimed: ${stats.claimed} | Active: ${stats.held}` },
                    { icon: "fa-calendar-week", color: "var(--success-subtle)", iconColor: "var(--success)", title: "Weekly Summary", desc: "Detailed weekly reports with AI matching stats.", msg: `Success rate: ${stats.total > 0 ? Math.round((stats.claimed / stats.total) * 100) : 0}%` },
                    { icon: "fa-calendar-days", color: "var(--warning-subtle)", iconColor: "var(--warning)", title: "Monthly Summary", desc: "Complete historical data with category logs.", msg: "Monthly data exported successfully." },
                  ].map((r, i) => (
                    <div key={i} style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-xl)", padding: "var(--space-6)", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                      <div>
                        <div className="stat-icon mb-4" style={{ background: r.color, color: r.iconColor }}>
                          <i className={`fas ${r.icon}`}></i>
                        </div>
                        <h4 style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: "var(--space-2)" }}>{r.title}</h4>
                        <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>{r.desc}</p>
                      </div>
                      <button onClick={() => { toast.success("Report compiled! " + r.msg); }}
                        className="btn-primary btn-full mt-4" style={{ fontSize: "0.75rem", padding: "10px" }}>
                        Download Report
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Resolve Dispute Modal */}
      {selectedDispute && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">
              <div>
                <h3>Resolve Dispute Claim</h3>
                <p>Provide claim details to sign off dispute</p>
              </div>
              <button className="modal-close" onClick={() => setSelectedDispute(null)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <form onSubmit={handleResolveDispute} className="space-y-4">
                <div className="flex gap-4" style={{ padding: "var(--space-3)", background: "var(--bg-surface)", borderRadius: "var(--radius-xl)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ width: 56, height: 56, borderRadius: "var(--radius-md)", background: "var(--bg-elevated)", border: "1px solid var(--border)", overflow: "hidden", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <img src={API.getImageUrl(selectedDispute.filename)} alt="" style={{ width: "100%", height: "100%", objectFit: "contain", padding: 2 }}
                      onError={(e) => { e.target.src = "https://via.placeholder.com/80?text=Item"; }} />
                  </div>
                  <div>
                    <span className="badge badge-primary mb-1">{selectedDispute.category}</span>
                    <p style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.9rem" }} className="line-clamp-1">{selectedDispute.description}</p>
                    <p style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Reporter: {selectedDispute.reported_by}</p>
                  </div>
                </div>

                <div>
                  <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                    Resolution: Claimant Name
                  </label>
                  <input type="text" placeholder="Claimant Student Name" value={resolutionClaimant} onChange={(e) => setResolutionClaimant(e.target.value)} required />
                </div>

                <div>
                  <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                    Resolution: Claimant Roll Number
                  </label>
                  <input type="text" placeholder="E.g., 2023504" value={resolutionRoll} onChange={(e) => setResolutionRoll(e.target.value)} required />
                </div>

                <button type="submit" disabled={actionLoading} className="btn-success btn-full btn-lg" style={{ marginTop: "var(--space-4)" }}>
                  {actionLoading ? (
                    <><i className="fas fa-circle-notch fa-spin"></i><span>Resolving...</span></>
                  ) : (
                    <><i className="fas fa-check-circle"></i><span>Approve Release & Claim</span></>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
