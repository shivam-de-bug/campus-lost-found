import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import API from "../api/apiClient";
import Header from "../components/Header";
import SearchBar from "../components/SearchBar";
import ReportForm from "../components/ReportForm";
import ItemCard from "../components/ItemCard";

export default function Home() {
  const [activeTab, setActiveTab] = useState("found"); // found, lost, my-reports
  const [foundItems, setFoundItems] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showReportForm, setShowReportForm] = useState(false);
  
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
    if (!file) {
      alert("Please select an image");
      return;
    }

    setLoading(true);
    try {
      const results = await API.searchLost({ file });
      setSearchResults(results.matches || []);
    } catch (error) {
      alert("Error searching: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchByText = async (text) => {
    if (!text.trim()) {
      alert("Please enter a search term");
      return;
    }

    setLoading(true);
    try {
      const results = await API.searchLost({ text_query: text });
      setSearchResults(results.matches || []);
    } catch (error) {
      alert("Error searching: " + error.message);
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
      alert("Item reported successfully!");
      setShowReportForm(false);
      loadFoundItems();
    } catch (error) {
      alert("Error reporting item: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFlagDispute = async (filename) => {
    if (!confirm("Are you sure you want to flag this claim handover as disputed? Admins will review the verified logs.")) {
      return;
    }
    setLoading(true);
    try {
      await API.updateItemStatus(filename, "disputed");
      alert("Dispute flagged successfully! Administrators have been notified.");
      loadFoundItems();
    } catch (error) {
      alert("Error flagging dispute: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Filter items reported by current logged-in user (student)
  const myReportedItems = foundItems.filter(
    (item) => item.reported_by === currentUser.email
  );

  return (
    <div className="min-h-screen bg-slate-50 pb-20 animate-slide-up">
      <Header />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Report Form Tab */}
        {activeTab === "report" && (
          <div className="bg-white rounded-3xl border border-slate-200/80 p-6 mb-8 max-w-2xl mx-auto">
            <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center">
              <i className="fas fa-plus-circle text-indigo-600 mr-2"></i>Log a Newly Found Item
            </h3>
            <ReportForm
              onSubmit={handleReportFound}
              loading={loading}
              onCancel={() => setShowReportForm(false)}
            />
          </div>
        )}

        {/* Browse Found Items Tab */}
        {activeTab === "found" && (
          <div>
            <h2 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2">
              <i className="fas fa-boxes-stacked text-indigo-600"></i>Recently Found on Campus
            </h2>
            {loading ? (
              <div className="text-center py-16">
                <i className="fas fa-spinner fa-spin text-4xl text-indigo-600 mb-3"></i>
                <p className="text-slate-500 font-medium">Scanning campus inventory...</p>
              </div>
            ) : foundItems.length === 0 ? (
              <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center max-w-lg mx-auto">
                <i className="fas fa-box-open text-6xl text-slate-300 mb-4"></i>
                <h3 className="text-xl font-bold text-slate-700">No Found Items</h3>
                <p className="text-slate-500 mt-2">
                  There are currently no unclaimed items in the system. Check back later or file a search query.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {foundItems.map((item, idx) => (
                  <ItemCard key={idx} item={item} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Tab (SigLIP matching integration) */}
        {activeTab === "lost" && (
          <div>
            <SearchBar
              onImageSearch={handleSearchByImage}
              onTextSearch={handleSearchByText}
              loading={loading}
            />

            {/* Results Gallery */}
            {searchResults.length > 0 && (
              <div className="mt-8">
                <h3 className="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2">
                  <i className="fas fa-circle-nodes text-emerald-500"></i>AI Matching Results
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {searchResults.map((item, idx) => (
                    <div key={idx} className="relative group">
                      <ItemCard item={item} />
                      <div className={`absolute top-4 right-4 text-xs font-bold px-3 py-1 rounded-full shadow-lg ${
                        item.confidence === "High"
                          ? "bg-emerald-600 text-white"
                          : item.confidence === "Medium"
                          ? "bg-amber-500 text-white"
                          : "bg-slate-600 text-white"
                      }`}>
                        {item.confidence} Match ({Math.round(item.similarity * 100)}%)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && searchResults.length === 0 && (
              <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center max-w-lg mx-auto mt-6">
                <i className="fas fa-wand-magic-sparkles text-6xl text-indigo-200 mb-4"></i>
                <h3 className="text-xl font-bold text-slate-700">Multimodal AI Search</h3>
                <p className="text-slate-500 mt-2">
                  Upload an image of your lost item or describe it in natural language. The backend SigLIP vision-language neural network will calculate vector embeddings to find matching items.
                </p>
              </div>
            )}
          </div>
        )}

        {/* My Reported Items Tab */}
        {activeTab === "my-reports" && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6">
            <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
              <i className="fas fa-clipboard-list text-indigo-600"></i>My Reports Registry
            </h3>
            {myReportedItems.length === 0 ? (
              <div className="text-center py-12">
                <i className="fas fa-file-circle-exclamation text-5xl text-slate-300 mb-3"></i>
                <p className="text-slate-500 font-medium">You have not reported any items yet.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse text-left">
                  <thead>
                    <tr className="bg-slate-50 text-slate-600 border-b border-slate-200">
                      <th className="px-6 py-4 font-semibold">Photo</th>
                      <th className="px-6 py-4 font-semibold">Description</th>
                      <th className="px-6 py-4 font-semibold">Found Location</th>
                      <th className="px-6 py-4 font-semibold">Date Logged</th>
                      <th className="px-6 py-4 font-semibold">Status</th>
                      <th className="px-6 py-4 font-semibold text-center">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {myReportedItems.map((item, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/50 transition">
                        <td className="px-6 py-3">
                          <img
                            src={API.getImageUrl(item.filename)}
                            alt=""
                            className="w-12 h-12 rounded-lg object-cover border border-slate-200"
                            onError={(e) => { e.target.src = "https://via.placeholder.com/50?text=Item"; }}
                          />
                        </td>
                        <td className="px-6 py-3 font-semibold text-slate-800">{item.description || "Found Item"}</td>
                        <td className="px-6 py-3 text-slate-600">{item.location}</td>
                        <td className="px-6 py-3 text-slate-500">
                          {item.timestamp ? new Date(item.timestamp * 1000).toLocaleDateString() : "Recently"}
                        </td>
                        <td className="px-6 py-3">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-bold uppercase ${
                            item.status === "claimed"
                              ? "bg-emerald-50 text-emerald-700"
                              : item.status === "disputed"
                              ? "bg-rose-50 text-rose-700"
                              : "bg-indigo-50 text-indigo-700"
                          }`}>
                            {item.status || "held"}
                          </span>
                        </td>
                        <td className="px-6 py-3 text-center">
                          {item.status === "claimed" && (
                            <button
                              onClick={() => handleFlagDispute(item.filename)}
                              className="bg-rose-50 hover:bg-rose-100 text-rose-600 px-3 py-1.5 rounded-lg text-xs font-bold transition border border-rose-100"
                            >
                              <i className="fas fa-triangle-exclamation mr-1"></i>Dispute Handover
                            </button>
                          )}
                          {(item.status === "held" || !item.status) && (
                            <span className="text-slate-400 text-xs font-medium">Held at Gates - safe</span>
                          )}
                          {item.status === "disputed" && (
                            <span className="text-rose-500 text-xs font-bold">Dispute Under Review</span>
                          )}
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

      {/* Premium Mobile Bottom Navigation Bar */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-lg border-t border-slate-200/80 shadow-[0_-4px_20px_rgba(0,0,0,0.03)] z-40 flex justify-around items-center h-16 px-4 pb-safe-bottom">
        <button
          onClick={() => { setActiveTab("found"); setSearchResults([]); }}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "found" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "found" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-house text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Browse</span>
        </button>

        <button
          onClick={() => { setActiveTab("lost"); setSearchResults([]); }}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "lost" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "lost" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-magnifying-glass text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">AI Search</span>
        </button>

        {/* Central Floating Action button */}
        <button
          onClick={() => setActiveTab("report")}
          className="flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 -mt-4 py-1"
        >
          <div className={`w-12 h-12 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center shadow-lg shadow-indigo-600/30 border-4 border-slate-50 hover:scale-105 transition-all duration-300 ${
            activeTab === "report" ? "from-indigo-600 to-purple-600" : ""
          }`}>
            <i className="fas fa-plus text-base"></i>
          </div>
          <span className={`text-[9px] font-bold mt-1 uppercase tracking-wider ${
            activeTab === "report" ? "text-indigo-600" : "text-slate-400"
          }`}>Report</span>
        </button>

        <button
          onClick={() => { setActiveTab("my-reports"); setSearchResults([]); }}
          className={`flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 ${
            activeTab === "my-reports" ? "text-indigo-600" : "text-slate-400"
          }`}
        >
          {activeTab === "my-reports" && <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>}
          <i className="fas fa-clipboard-list text-base"></i>
          <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">My Reports</span>
        </button>

        {currentUser.role && currentUser.role !== "student" && (
          <Link
            to={currentUser.role === "admin" ? "/admin" : "/guard"}
            className="flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1 text-slate-400 hover:text-indigo-600 no-underline"
          >
            <i className="fas fa-shield-halved text-base"></i>
            <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Control</span>
          </Link>
        )}
      </nav>
    </div>
  );
}
