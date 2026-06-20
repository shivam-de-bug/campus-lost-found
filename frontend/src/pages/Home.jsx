import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import API from "../api/apiClient";
import { useToast } from "../components/ToastContext";
import Header from "../components/Header";
import SearchBar from "../components/SearchBar";
import ReportForm from "../components/ReportForm";
import ItemCard from "../components/ItemCard";
import { SkeletonGrid } from "../components/Skeleton";

export default function Home() {
  const toast = useToast();
  const [activeTab, setActiveTab] = useState("found");
  const [foundItems, setFoundItems] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const currentUser = API.getCurrentUser() || { email: "anonymous" };

  useEffect(() => {
    loadFoundItems();
  }, []);

  const loadFoundItems = async () => {
    setLoading(true);
    try {
      const data = await API.getAllFound();
      setFoundItems(data.items || []);
    } catch (error) {
      console.error("Error loading items:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchByImage = async (file) => {
    if (!file) { toast.warning("Please select an image"); return; }
    setLoading(true);
    try {
      const results = await API.searchLost({ file });
      setSearchResults(results.matches || []);
      if ((results.matches || []).length > 0) {
        toast.success(`Found ${results.matches.length} matching items!`);
      } else {
        toast.info("No matches found. Try a different image.");
      }
    } catch (error) {
      toast.error("Error searching: " + error.message);
    } finally { setLoading(false); }
  };

  const handleSearchByText = async (text) => {
    if (!text.trim()) { toast.warning("Please enter a search term"); return; }
    setLoading(true);
    try {
      const results = await API.searchLost({ text_query: text });
      setSearchResults(results.matches || []);
      if ((results.matches || []).length > 0) {
        toast.success(`Found ${results.matches.length} matching items!`);
      } else {
        toast.info("No matches found. Try different keywords.");
      }
    } catch (error) {
      toast.error("Error searching: " + error.message);
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
      toast.success("Item reported successfully!");
      setActiveTab("found");
      loadFoundItems();
    } catch (error) {
      toast.error("Error reporting item: " + error.message);
    } finally { setLoading(false); }
  };

  const handleFlagDispute = async (filename) => {
    if (!confirm("Are you sure you want to flag this claim as disputed?")) return;
    setLoading(true);
    try {
      await API.updateItemStatus(filename, "disputed");
      toast.success("Dispute flagged successfully! Administrators notified.");
      loadFoundItems();
    } catch (error) {
      toast.error("Error flagging dispute: " + error.message);
    } finally { setLoading(false); }
  };

  const myReportedItems = foundItems.filter((item) => item.reported_by === currentUser.email);

  return (
    <div className="page animate-fade-in">
      <Header />

      <main className="page-content">
        {/* Welcome Hero */}
        <div className="animate-slide-up mb-8" style={{ position: "relative" }}>
          <div className="bg-blob" style={{
            width: 200, height: 200, background: "rgba(124,58,237,0.06)", top: -50, right: -50, filter: "blur(60px)", position: "absolute"
          }}></div>
          <h2 className="gradient-text" style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: "clamp(1.5rem, 4vw, 2rem)",
            fontWeight: 800,
            margin: 0,
          }}>
            Welcome back, {currentUser.name || "Explorer"} 👋
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "var(--space-2)" }}>
            {foundItems.length} items currently in campus inventory
          </p>
        </div>

        {/* Report Form Tab */}
        {activeTab === "report" && (
          <div className="max-w-2xl mx-auto mb-8">
            <ReportForm
              onSubmit={handleReportFound}
              loading={loading}
              onCancel={() => setActiveTab("found")}
            />
          </div>
        )}

        {/* Browse Found Items Tab */}
        {activeTab === "found" && (
          <div className="animate-slide-up">
            <div className="section-header">
              <i className="fas fa-boxes-stacked"></i>
              <h2>Recently Found on Campus</h2>
            </div>
            {loading ? (
              <SkeletonGrid count={4} />
            ) : foundItems.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon"><i className="fas fa-box-open"></i></div>
                <h3>No Found Items</h3>
                <p>There are currently no unclaimed items in the system. Check back later or file a search query.</p>
              </div>
            ) : (
              <div className="grid grid-4">
                {foundItems.map((item, idx) => (
                  <ItemCard key={idx} item={item} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Tab */}
        {activeTab === "lost" && (
          <div className="animate-slide-up">
            <SearchBar
              onImageSearch={handleSearchByImage}
              onTextSearch={handleSearchByText}
              loading={loading}
            />

            {searchResults.length > 0 && (
              <div className="mt-6">
                <div className="section-header">
                  <i className="fas fa-circle-nodes text-success"></i>
                  <h3>AI Matching Results</h3>
                  <span className="badge badge-success" style={{ marginLeft: "auto" }}>
                    {searchResults.length} matches
                  </span>
                </div>
                <div className="grid grid-4">
                  {searchResults.map((item, idx) => (
                    <div key={idx} className="relative">
                      <ItemCard item={item} />
                      <div className="match-badge">
                        <span className={`badge ${
                          item.confidence === "High" ? "badge-success"
                            : item.confidence === "Medium" ? "badge-warning"
                            : "badge-primary"
                        }`} style={{ fontSize: "0.6rem" }}>
                          {item.confidence} ({Math.round(item.similarity * 100)}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && searchResults.length === 0 && (
              <div className="empty-state mt-6">
                <div className="empty-state-icon"><i className="fas fa-wand-magic-sparkles"></i></div>
                <h3>Multimodal AI Search</h3>
                <p>Upload an image of your lost item or describe it in natural language. The SigLIP neural network will find matching items.</p>
              </div>
            )}
          </div>
        )}

        {/* My Reported Items Tab */}
        {activeTab === "my-reports" && (
          <div className="animate-slide-up">
            <div className="card">
              <div className="card-body" style={{ padding: "var(--space-6)" }}>
                <div className="section-header" style={{ marginBottom: "var(--space-5)" }}>
                  <i className="fas fa-clipboard-list"></i>
                  <h3>My Reports Registry</h3>
                </div>
                {myReportedItems.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon"><i className="fas fa-file-circle-exclamation"></i></div>
                    <p style={{ fontWeight: 600 }}>You have not reported any items yet.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Photo</th>
                          <th>Description</th>
                          <th>Found Location</th>
                          <th>Date Logged</th>
                          <th>Status</th>
                          <th style={{ textAlign: "center" }}>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {myReportedItems.map((item, idx) => (
                          <tr key={idx}>
                            <td>
                              <img
                                src={API.getImageUrl(item.filename)}
                                alt=""
                                style={{ width: 44, height: 44, borderRadius: "var(--radius-md)", objectFit: "cover", border: "1px solid var(--border)" }}
                                onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Item"; }}
                              />
                            </td>
                            <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                              {item.description || "Found Item"}
                            </td>
                            <td>{item.location}</td>
                            <td>{item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Recently"}</td>
                            <td>
                              <span className={`badge ${
                                item.status === "claimed" ? "badge-success"
                                  : item.status === "disputed" ? "badge-danger"
                                  : "badge-primary"
                              }`}>
                                {item.status || "held"}
                              </span>
                            </td>
                            <td style={{ textAlign: "center" }}>
                              {item.status === "claimed" && (
                                <button onClick={() => handleFlagDispute(item.filename)} className="btn-danger" style={{ fontSize: "0.7rem", padding: "4px 10px" }}>
                                  <i className="fas fa-triangle-exclamation"></i>Dispute
                                </button>
                              )}
                              {(item.status === "held" || !item.status) && (
                                <span style={{ color: "var(--text-muted)", fontSize: "0.7rem", fontWeight: 500 }}>Held at Gates</span>
                              )}
                              {item.status === "disputed" && (
                                <span style={{ color: "var(--danger)", fontSize: "0.7rem", fontWeight: 700 }}>Under Review</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Bottom Navigation */}
      <nav className="bottom-nav">
        <button
          onClick={() => { setActiveTab("found"); setSearchResults([]); }}
          className={`nav-item ${activeTab === "found" ? "active" : ""}`}
        >
          <i className="fas fa-house"></i>
          <span>Browse</span>
        </button>

        <button
          onClick={() => { setActiveTab("lost"); setSearchResults([]); }}
          className={`nav-item ${activeTab === "lost" ? "active" : ""}`}
        >
          <i className="fas fa-magnifying-glass"></i>
          <span>AI Search</span>
        </button>

        {/* Central FAB */}
        <button
          onClick={() => setActiveTab("report")}
          className={`nav-fab ${activeTab === "report" ? "active" : ""}`}
        >
          <div className="nav-fab-circle">
            <i className="fas fa-plus"></i>
          </div>
          <span>Report</span>
        </button>

        <button
          onClick={() => { setActiveTab("my-reports"); setSearchResults([]); }}
          className={`nav-item ${activeTab === "my-reports" ? "active" : ""}`}
        >
          <i className="fas fa-clipboard-list"></i>
          <span>My Reports</span>
        </button>

        {currentUser.role && currentUser.role !== "student" && (
          <Link
            to={currentUser.role === "admin" ? "/admin" : "/guard"}
            className="nav-item"
            style={{ textDecoration: "none" }}
          >
            <i className="fas fa-shield-halved"></i>
            <span>Control</span>
          </Link>
        )}
      </nav>
    </div>
  );
}
