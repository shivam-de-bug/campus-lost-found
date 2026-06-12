import { useState, useEffect } from "react";
import API from "../api/apiClient";

export default function Admin() {
  const [activeSection, setActiveSection] = useState("dashboard");
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    total: 0,
    claimed: 0,
    activeUsers: 0,
    matchRate: 92,
  });

  useEffect(() => {
    loadDashboardData();
  }, []);

  useEffect(() => {
    if (activeSection === "items") {
      loadItemsData();
    }
  }, [activeSection]);

  const loadDashboardData = async () => {
    setLoading(true);
    const data = await API.getAllFound();
    setItems(data.items || []);
    setStats({
      total: data.items?.length || 0,
      claimed: Math.floor((data.items?.length || 0) * 0.3),
      activeUsers: Math.floor(Math.random() * 150) + 50,
      matchRate: 92,
    });
    setLoading(false);
  };

  const loadItemsData = async () => {
    setLoading(true);
    const data = await API.getAllFound();
    setItems(data.items || []);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Sidebar */}
      <aside className="w-14 bg-white shadow-lg">
        <nav className="flex flex-col items-center py-4 gap-2">
          <button
            onClick={() => setActiveSection("dashboard")}
            title="Dashboard"
            className={`w-12 h-12 rounded-lg flex items-center justify-center transition ${
              activeSection === "dashboard"
                ? "bg-blue-50 text-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-chart-line text-xl"></i>
          </button>
          <button
            onClick={() => setActiveSection("items")}
            title="Items"
            className={`w-12 h-12 rounded-lg flex items-center justify-center transition ${
              activeSection === "items"
                ? "bg-blue-50 text-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-box text-xl"></i>
          </button>
          <button
            onClick={() => setActiveSection("disputes")}
            title="Disputes"
            className={`w-12 h-12 rounded-lg flex items-center justify-center transition ${
              activeSection === "disputes"
                ? "bg-blue-50 text-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-gavel text-xl"></i>
          </button>
          <button
            onClick={() => setActiveSection("analytics")}
            title="Analytics"
            className={`w-12 h-12 rounded-lg flex items-center justify-center transition ${
              activeSection === "analytics"
                ? "bg-blue-50 text-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-chart-bar text-xl"></i>
          </button>
          <button
            onClick={() => setActiveSection("reports")}
            title="Reports"
            className={`w-12 h-12 rounded-lg flex items-center justify-center transition ${
              activeSection === "reports"
                ? "bg-blue-50 text-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            <i className="fas fa-file-alt text-xl"></i>
          </button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1">
        {/* Header */}
        <header className="bg-white shadow-sm sticky top-0 z-40">
          <div className="px-6 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold text-blue-600">
              <i className="fas fa-shield mr-2"></i>Admin Dashboard
            </h1>
            <a
              href="/"
              className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600"
            >
              Logout
            </a>
          </div>
        </header>

        {/* Content Area */}
        <div className="p-6">
          {/* Dashboard Section */}
          {activeSection === "dashboard" && (
            <div>
              {/* Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-gray-600 text-sm">Total Items</p>
                      <p className="text-3xl font-bold text-blue-600">
                        {stats.total}
                      </p>
                    </div>
                    <i className="fas fa-box text-3xl text-blue-100"></i>
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-gray-600 text-sm">Claimed</p>
                      <p className="text-3xl font-bold text-green-600">
                        {stats.claimed}
                      </p>
                    </div>
                    <i className="fas fa-check-circle text-3xl text-green-100"></i>
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-gray-600 text-sm">Active Users</p>
                      <p className="text-3xl font-bold text-purple-600">
                        {stats.activeUsers}
                      </p>
                    </div>
                    <i className="fas fa-users text-3xl text-purple-100"></i>
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-gray-600 text-sm">AI Match Rate</p>
                      <p className="text-3xl font-bold text-amber-600">
                        {stats.matchRate}%
                      </p>
                    </div>
                    <i className="fas fa-brain text-3xl text-amber-100"></i>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-bold mb-4">
                  <i className="fas fa-history mr-2"></i>Recent Activity
                </h2>
                {loading ? (
                  <p>Loading...</p>
                ) : items.length === 0 ? (
                  <p className="text-gray-500">No recent items</p>
                ) : (
                  <div className="space-y-4">
                    {items.slice(0, 5).map((item, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100"
                      >
                        <div className="flex-1">
                          <p className="font-semibold">
                            {item.description || "New Item"}
                          </p>
                          <p className="text-sm text-gray-600">
                            Location: {item.location}
                          </p>
                        </div>
                        <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                          New
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Items Section */}
          {activeSection === "items" && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">
                <i className="fas fa-box mr-2"></i>Items Management
              </h2>
              {loading ? (
                <p>Loading...</p>
              ) : items.length === 0 ? (
                <p className="text-gray-500">No items found</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                          Description
                        </th>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                          Location
                        </th>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                          Category
                        </th>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                          Contact
                        </th>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item, idx) => (
                        <tr
                          key={idx}
                          className="border-t hover:bg-gray-50"
                        >
                          <td className="px-6 py-4 text-sm">
                            {item.description || "N/A"}
                          </td>
                          <td className="px-6 py-4 text-sm">
                            {item.location}
                          </td>
                          <td className="px-6 py-4 text-sm">
                            <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs">
                              {item.category || "Uncategorized"}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm">
                            {item.contact}
                          </td>
                          <td className="px-6 py-4 text-sm">
                            <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs">
                              Active
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

          {/* Disputes Section */}
          {activeSection === "disputes" && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">
                <i className="fas fa-gavel mr-2"></i>Dispute Resolution
              </h2>
              <div className="text-center py-12">
                <i className="fas fa-inbox text-4xl text-gray-300 mb-4"></i>
                <p className="text-gray-500">
                  No disputes at this time. Great work!
                </p>
              </div>
            </div>
          )}

          {/* Analytics Section */}
          {activeSection === "analytics" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-bold mb-4">
                  <i className="fas fa-chart-pie mr-2"></i>Items by Category
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>Electronics</span>
                    <div className="w-48 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: "45%" }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Accessories</span>
                    <div className="w-48 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{ width: "30%" }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Documents</span>
                    <div className="w-48 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-yellow-600 h-2 rounded-full"
                        style={{ width: "25%" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-bold mb-4">
                  <i className="fas fa-chart-line mr-2"></i>Weekly Trend
                </h3>
                <p className="text-gray-500">
                  Analytics data visualization would be displayed here
                </p>
              </div>
            </div>
          )}

          {/* Reports Section */}
          {activeSection === "reports" && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">
                <i className="fas fa-file-alt mr-2"></i>Generate Report
              </h2>
              <div className="space-y-4">
                <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700">
                  Daily Report
                </button>
                <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700">
                  Weekly Report
                </button>
                <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700">
                  Monthly Report
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
