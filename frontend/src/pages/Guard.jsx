import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import API from "../api/apiClient";
import Header from "../components/Header";
import ReportForm from "../components/ReportForm";

export default function Guard() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("inventory"); // inventory, intake, logs
  const [selectedItem, setSelectedItem] = useState(null); // for claiming modal
  
  // Claim form states
  const [claimantName, setClaimantName] = useState("");
  const [claimantRoll, setClaimantRoll] = useState("");
  const [verifiedId, setVerifiedId] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  // Stats
  const [stats, setStats] = useState({ held: 0, claimed: 0 });

  useEffect(() => {
    loadInventory();
  }, []);

  const loadInventory = async () => {
    setLoading(true);
    try {
      const data = await API.getAllFound();
      const allItems = data.items || [];
      setItems(allItems);
      
      const heldCount = allItems.filter(i => i.status === "held").length;
      const claimedCount = allItems.filter(i => i.status === "claimed").length;
      setStats({ held: heldCount, claimed: claimedCount });
    } catch (error) {
      console.error("Error loading items:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleReportFound = async (formData, reportFile) => {
    if (!reportFile || !formData.location || !formData.contact) {
      alert("Please fill all required fields");
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
      alert("Item logged successfully at gate inventory!");
      setActiveTab("inventory");
      loadInventory();
    } catch (error) {
      alert("Error reporting item: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleProcessClaim = async (e) => {
    e.preventDefault();
    if (!claimantName || !claimantRoll || !verifiedId) {
      alert("Please complete the verification checks");
      return;
    }

    setActionLoading(true);
    try {
      await API.updateItemStatus(selectedItem.filename, "claimed", claimantRoll, claimantName);
      alert("Item status updated to CLAIMED successfully!");
      setSelectedItem(null);
      setClaimantName("");
      setClaimantRoll("");
      setVerifiedId(false);
      loadInventory();
    } catch (error) {
      alert("Error processing claim: " + error.message);
    } finally {
      setActionLoading(false);
    }
  };

  const activeHeldItems = items.filter(item => item.status === "held" || !item.status);
  const claimedHistory = items.filter(item => item.status === "claimed");

  return (
    <div className="min-h-screen bg-slate-50 pb-20 animate-slide-up">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Page title and description */}
        <div className="mb-8 flex justify-between items-center flex-wrap gap-4">
          <div>
            <h2 className="text-3xl font-bold text-slate-800 m-0">
              <i className="fas fa-user-shield mr-2 text-indigo-600"></i>Security Staff Panel
            </h2>
            <p className="text-slate-500 text-sm mt-1">
              Campus Gate Inventory Management & Verified Handover
            </p>
          </div>
          <div className="flex gap-4">
            <div className="bg-white px-5 py-3 rounded-xl shadow-sm border border-slate-200 text-center">
              <span className="text-xs font-semibold text-slate-400 block uppercase">Held at Gates</span>
              <span className="text-2xl font-bold text-indigo-600">{stats.held}</span>
            </div>
            <div className="bg-white px-5 py-3 rounded-xl shadow-sm border border-slate-200 text-center">
              <span className="text-xs font-semibold text-slate-400 block uppercase">Claimed Handovers</span>
              <span className="text-2xl font-bold text-emerald-600">{stats.claimed}</span>
            </div>
          </div>
        </div>

        {/* Inventory View */}
        {activeTab === "inventory" && (
          <div>
            {loading ? (
              <div className="text-center py-16">
                <i className="fas fa-spinner fa-spin text-4xl text-indigo-600 mb-3"></i>
                <p className="text-slate-500 font-medium">Loading gate inventory...</p>
              </div>
            ) : activeHeldItems.length === 0 ? (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center max-w-lg mx-auto mt-6">
                <i className="fas fa-box-open text-6xl text-slate-300 mb-4"></i>
                <h3 className="text-xl font-bold text-slate-700">Empty Gate Inventory</h3>
                <p className="text-slate-500 mt-2">
                  There are currently no items held at any campus gates. Newly reported items will appear here.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {activeHeldItems.map((item, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition duration-300 flex flex-col justify-between"
                  >
                    <div>
                      <div className="relative h-48 bg-slate-100 overflow-hidden flex items-center justify-center border-b border-slate-100">
                        <img
                          src={API.getImageUrl(item.filename || "")}
                          onError={(e) => {
                            e.target.src = "https://via.placeholder.com/300?text=Gate+Intake";
                          }}
                          alt={item.description}
                          className="w-full h-full object-contain p-2 hover:scale-105 transition duration-500"
                        />
                      </div>
                      <div className="p-5">
                        <span className="bg-indigo-50 text-indigo-700 text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
                          {item.category || "General"}
                        </span>
                        <h3 className="font-bold text-lg text-slate-800 mt-3 mb-2 line-clamp-1">
                          {item.description || "Unidentified Item"}
                        </h3>
                        <p className="text-slate-500 text-sm flex items-center mb-1">
                          <i className="fas fa-map-marker-alt text-slate-400 w-5"></i>
                          {item.location}
                        </p>
                        <p className="text-slate-500 text-sm flex items-center mb-1">
                          <i className="fas fa-calendar text-slate-400 w-5"></i>
                          {item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Just now"}
                        </p>
                        <p className="text-slate-500 text-sm flex items-center">
                          <i className="fas fa-envelope text-slate-400 w-5"></i>
                          {item.contact}
                        </p>
                      </div>
                    </div>
                    <div className="p-5 pt-0">
                      <button
                        onClick={() => setSelectedItem(item)}
                        className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 rounded-xl transition duration-300 flex items-center justify-center gap-2"
                      >
                        <i className="fas fa-hand-holding-hand"></i>Process Claim
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* intake form tab */}
        {activeTab === "intake" && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center">
                <i className="fas fa-plus-circle text-indigo-600 mr-2"></i>Log Found Item at Gate
              </h3>
              <ReportForm
                onSubmit={handleReportFound}
                loading={loading}
                onCancel={() => setActiveTab("inventory")}
              />
            </div>
          </div>
        )}

        {/* Handover Logs Tab */}
        {activeTab === "logs" && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 overflow-hidden">
            <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center">
              <i className="fas fa-history text-indigo-600 mr-2"></i>Verified Claim logs
            </h3>
            {claimedHistory.length === 0 ? (
              <div className="text-center py-12">
                <i className="fas fa-folder-open text-5xl text-slate-300 mb-3"></i>
                <p className="text-slate-500 font-medium">No claimed items logged yet.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse text-left">
                  <thead>
                    <tr className="bg-slate-50 text-slate-600 border-b border-slate-200">
                      <th className="px-6 py-4 font-semibold">Item Photo</th>
                      <th className="px-6 py-4 font-semibold">Description</th>
                      <th className="px-6 py-4 font-semibold">Claimed By (Name / Roll)</th>
                      <th className="px-6 py-4 font-semibold">Handed Over By</th>
                      <th className="px-6 py-4 font-semibold">Date Handed Over</th>
                      <th className="px-6 py-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {claimedHistory.map((item, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/50 transition">
                        <td className="px-6 py-3">
                          <img
                            src={API.getImageUrl(item.filename)}
                            alt=""
                            className="w-12 h-12 rounded-lg object-cover border border-slate-200"
                            onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Img"; }}
                          />
                        </td>
                        <td className="px-6 py-3 font-semibold text-slate-800">
                          {item.description || "N/A"}
                          <div className="text-xs text-slate-400 font-normal">Found at: {item.location}</div>
                        </td>
                        <td className="px-6 py-3 text-slate-600">
                          <div className="font-semibold">{item.claimed_by_name || "N/A"}</div>
                          <div className="text-xs text-indigo-500 font-bold">{item.claimed_by || "N/A"}</div>
                        </td>
                        <td className="px-6 py-3 text-slate-500 text-xs font-mono">{item.handed_over_by || "System Admin"}</td>
                        <td className="px-6 py-3 text-slate-500">
                          {item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Recently"}
                        </td>
                        <td className="px-6 py-3">
                          <span className="bg-emerald-50 text-emerald-700 text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider border border-emerald-200">
                            Claimed
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Claim Handover Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl max-w-md w-full shadow-2xl border border-slate-100 overflow-hidden transform transition duration-300 animate-in fade-in zoom-in-95">
            <div className="bg-indigo-600 p-6 text-white flex justify-between items-center">
              <div>
                <h3 className="font-bold text-xl m-0">Verified Claim Handover</h3>
                <p className="text-indigo-100 text-xs mt-1">Please inspect claimant ID before releasing</p>
              </div>
              <button
                onClick={() => setSelectedItem(null)}
                className="text-white/80 hover:text-white bg-white/10 hover:bg-white/20 p-2 rounded-full transition"
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
            
            <form onSubmit={handleProcessClaim} className="p-6 space-y-4">
              <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border border-slate-100 mb-2">
                <div className="w-16 h-16 rounded-xl bg-slate-100 border border-slate-200 shadow-sm overflow-hidden flex items-center justify-center flex-shrink-0">
                  <img
                    src={API.getImageUrl(selectedItem.filename)}
                    alt=""
                    className="w-full h-full object-contain p-1"
                    onError={(e) => { e.target.src = "https://via.placeholder.com/80?text=Item"; }}
                  />
                </div>
                <div>
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{selectedItem.category}</span>
                  <p className="font-bold text-slate-800 line-clamp-1">{selectedItem.description}</p>
                  <p className="text-xs text-slate-500"><i className="fas fa-map-pin mr-1"></i>Held at: {selectedItem.location}</p>
                </div>
              </div>

              <div>
                <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-1.5">
                  Claimant Full Name
                </label>
                <input
                  type="text"
                  placeholder="Enter student name"
                  value={claimantName}
                  onChange={(e) => setClaimantName(e.target.value)}
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-1.5">
                  Claimant Roll Number
                </label>
                <input
                  type="text"
                  placeholder="E.g., 2023504"
                  value={claimantRoll}
                  onChange={(e) => setClaimantRoll(e.target.value)}
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20"
                  required
                />
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 flex gap-3 items-start mt-2">
                <input
                  type="checkbox"
                  id="verify_id"
                  checked={verifiedId}
                  onChange={(e) => setVerifiedId(e.target.checked)}
                  className="w-4 h-4 mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  required
                />
                <label htmlFor="verify_id" className="text-xs text-slate-600 leading-relaxed font-medium cursor-pointer select-none">
                  I confirm that I have verified the claimant's student identity card and details are fully correct.
                </label>
              </div>

              <button
                type="submit"
                disabled={actionLoading || !verifiedId}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 rounded-xl transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mt-4"
              >
                {actionLoading ? (
                  <>
                    <i className="fas fa-circle-notch fa-spin"></i>
                    <span>Processing Handover...</span>
                  </>
                ) : (
                  <>
                    <i className="fas fa-check-circle"></i>
                    <span>Release Item</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Premium Guard Bottom Navigation Bar */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-lg border-t border-slate-200/80 shadow-[0_-4px_20px_rgba(0,0,0,0.03)] z-40 flex justify-around items-center h-16 px-4 pb-safe-bottom">
        <button
          onClick={() => setActiveTab("inventory")}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "inventory" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "inventory" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-boxes-stacked text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Inventory</span>
        </button>

        <button
          onClick={() => setActiveTab("intake")}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "intake" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "intake" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-plus-circle text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Intake</span>
        </button>

        <button
          onClick={() => setActiveTab("logs")}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "logs" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "logs" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-clipboard-check text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Logs</span>
        </button>

        <button
          onClick={() => navigate("/")}
          className="flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 text-slate-400 hover:text-slate-600"
        >
          <i className="fas fa-circle-arrow-left text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Exit</span>
        </button>
      </nav>
    </div>
  );
}
