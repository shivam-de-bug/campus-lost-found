import { useState, useEffect } from "react";
import API from "../api/apiClient";

export default function Home() {
  const [activeTab, setActiveTab] = useState("found"); // 'found' or 'lost'
  const [foundItems, setFoundItems] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [searchMode, setSearchMode] = useState("image"); // 'image' or 'text'
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchText, setSearchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [showReportForm, setShowReportForm] = useState(false);
  const [reportFormData, setReportFormData] = useState({
    description: "",
    location: "",
    contact: "",
    category: "",
  });
  const [reportFile, setReportFile] = useState(null);

  // Load found items on component mount
  useEffect(() => {
    loadFoundItems();
  }, []);

  const loadFoundItems = async () => {
    setLoading(true);
    const data = await API.getAllFound();
    setFoundItems(data.items || []);
    setLoading(false);
  };

  const handleSearchByImage = async () => {
    if (!selectedFile) {
      alert("Please select an image");
      return;
    }

    setLoading(true);
    try {
      const results = await API.searchLost({ file: selectedFile });
      setSearchResults(results.matches || []);
    } catch (error) {
      alert("Error searching: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchByText = async () => {
    if (!searchText.trim()) {
      alert("Please enter a search term");
      return;
    }

    setLoading(true);
    try {
      const results = await API.searchLost({ text_query: searchText });
      setSearchResults(results.matches || []);
    } catch (error) {
      alert("Error searching: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReportFound = async () => {
    if (
      !reportFile ||
      !reportFormData.location ||
      !reportFormData.contact
    ) {
      alert("Please fill all required fields");
      return;
    }

    const formData = new FormData();
    formData.append("file", reportFile);
    formData.append("location", reportFormData.location);
    formData.append("contact", reportFormData.contact);
    formData.append("description", reportFormData.description);
    formData.append("category", reportFormData.category);

    setLoading(true);
    try {
      await API.reportFound(formData);
      alert("Item reported successfully!");
      setShowReportForm(false);
      setReportFormData({
        description: "",
        location: "",
        contact: "",
        category: "",
      });
      setReportFile(null);
      loadFoundItems();
    } catch (error) {
      alert("Error reporting item: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-blue-600">
            <i className="fas fa-search mr-2"></i>404 Found
          </h1>
          <div className="flex gap-2">
            <a href="/admin" className="btn bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              Admin
            </a>
            <a href="/guard" className="btn bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700">
              Guard
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Tabs */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => {
              setActiveTab("found");
              setSearchResults([]);
            }}
            className={`px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === "found"
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-eye mr-2"></i>Browse Found Items
          </button>
          <button
            onClick={() => {
              setActiveTab("lost");
              setSearchResults([]);
            }}
            className={`px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === "lost"
                ? "bg-emerald-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-search mr-2"></i>Search Lost Items
          </button>
          <button
            onClick={() => setShowReportForm(!showReportForm)}
            className="px-6 py-3 rounded-lg font-semibold transition bg-amber-600 text-white hover:bg-amber-700"
          >
            <i className="fas fa-plus mr-2"></i>Report Found Item
          </button>
        </div>

        {/* Report Form */}
        {showReportForm && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-2xl font-bold mb-4">Report a Found Item</h2>
            <div className="space-y-4">
              <input
                type="text"
                placeholder="Item Description"
                value={reportFormData.description}
                onChange={(e) =>
                  setReportFormData({
                    ...reportFormData,
                    description: e.target.value,
                  })
                }
                className="w-full border rounded-lg px-4 py-2"
              />
              <input
                type="text"
                placeholder="Location Found *"
                value={reportFormData.location}
                onChange={(e) =>
                  setReportFormData({
                    ...reportFormData,
                    location: e.target.value,
                  })
                }
                className="w-full border rounded-lg px-4 py-2"
              />
              <input
                type="text"
                placeholder="Category (e.g., electronics, accessories)"
                value={reportFormData.category}
                onChange={(e) =>
                  setReportFormData({
                    ...reportFormData,
                    category: e.target.value,
                  })
                }
                className="w-full border rounded-lg px-4 py-2"
              />
              <input
                type="email"
                placeholder="Contact Email *"
                value={reportFormData.contact}
                onChange={(e) =>
                  setReportFormData({
                    ...reportFormData,
                    contact: e.target.value,
                  })
                }
                className="w-full border rounded-lg px-4 py-2"
              />
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setReportFile(e.target.files[0])}
                className="w-full border rounded-lg px-4 py-2"
              />
              <button
                onClick={handleReportFound}
                disabled={loading}
                className="w-full bg-amber-600 text-white py-2 rounded-lg hover:bg-amber-700 disabled:opacity-50"
              >
                {loading ? "Reporting..." : "Report Item"}
              </button>
            </div>
          </div>
        )}

        {/* Found Items Tab */}
        {activeTab === "found" && (
          <div>
            <h2 className="text-2xl font-bold mb-4">
              <i className="fas fa-box mr-2"></i>Recently Found Items
            </h2>
            {loading ? (
              <p className="text-center text-gray-500">Loading...</p>
            ) : foundItems.length === 0 ? (
              <p className="text-center text-gray-500">
                No found items yet
              </p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {foundItems.map((item, idx) => (
                  <div key={idx} className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition">
                    <img
                      src={API.getImageUrl(item.filename || "")}
                      onError={(e) =>
                        (e.target.src = "https://via.placeholder.com/300?text=Item")
                      }
                      alt={item.description}
                      className="w-full h-48 object-cover"
                    />
                    <div className="p-4">
                      <p className="font-semibold text-lg">
                        {item.description || "Found Item"}
                      </p>
                      <p className="text-gray-600">
                        <i className="fas fa-map-pin mr-2"></i>
                        {item.location}
                      </p>
                      <p className="text-gray-600">
                        <i className="fas fa-tag mr-2"></i>
                        {item.category || "Uncategorized"}
                      </p>
                      <p className="text-sm text-gray-500">
                        <i className="fas fa-envelope mr-2"></i>
                        {item.contact}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Lost Items Tab */}
        {activeTab === "lost" && (
          <div>
            <h2 className="text-2xl font-bold mb-4">
              <i className="fas fa-magnifying-glass mr-2"></i>Search for Your Lost Item
            </h2>

            {/* Search Mode Toggle */}
            <div className="flex gap-4 mb-6">
              <button
                onClick={() => setSearchMode("image")}
                className={`px-6 py-3 rounded-lg font-semibold transition ${
                  searchMode === "image"
                    ? "bg-blue-600 text-white"
                    : "bg-white text-gray-700 hover:bg-gray-50"
                }`}
              >
                <i className="fas fa-image mr-2"></i>Search by Image
              </button>
              <button
                onClick={() => setSearchMode("text")}
                className={`px-6 py-3 rounded-lg font-semibold transition ${
                  searchMode === "text"
                    ? "bg-blue-600 text-white"
                    : "bg-white text-gray-700 hover:bg-gray-50"
                }`}
              >
                <i className="fas fa-keyboard mr-2"></i>Search by Text
              </button>
            </div>

            {/* Search Input */}
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              {searchMode === "image" ? (
                <div>
                  <label className="block text-lg font-semibold mb-2">
                    Upload an image of your lost item
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    className="w-full border-2 border-dashed rounded-lg p-6 mb-4"
                  />
                  {selectedFile && (
                    <p className="text-green-600 mb-4">
                      <i className="fas fa-check-circle mr-2"></i>
                      {selectedFile.name}
                    </p>
                  )}
                  <button
                    onClick={handleSearchByImage}
                    disabled={loading || !selectedFile}
                    className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {loading ? "Searching..." : "Search by Image"}
                  </button>
                </div>
              ) : (
                <div>
                  <label className="block text-lg font-semibold mb-2">
                    Describe your lost item
                  </label>
                  <input
                    type="text"
                    placeholder="E.g., Black laptop, iPhone 12, house keys..."
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    className="w-full border rounded-lg px-4 py-2 mb-4"
                  />
                  <button
                    onClick={handleSearchByText}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {loading ? "Searching..." : "Search"}
                  </button>
                </div>
              )}
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div>
                <h3 className="text-xl font-bold mb-4">
                  <i className="fas fa-star mr-2"></i>Possible Matches ({searchResults.length})
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {searchResults.map((item, idx) => (
                    <div key={idx} className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition border-2 border-yellow-400">
                      <img
                        src={API.getImageUrl(item.filename || "")}
                        onError={(e) =>
                          (e.target.src = "https://via.placeholder.com/300?text=Match")
                        }
                        alt="Match"
                        className="w-full h-48 object-cover"
                      />
                      <div className="p-4">
                        <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full inline-block mb-2">
                          {Math.round((item.confidence || 0) * 100)}% Match
                        </div>
                        <p className="font-semibold text-lg">
                          {item.description || "Found Item"}
                        </p>
                        <p className="text-gray-600">
                          <i className="fas fa-map-pin mr-2"></i>
                          {item.location}
                        </p>
                        <p className="text-gray-600">
                          <i className="fas fa-tag mr-2"></i>
                          {item.category || "Uncategorized"}
                        </p>
                        <p className="text-sm text-blue-600">
                          <i className="fas fa-envelope mr-2"></i>
                          {item.contact}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
