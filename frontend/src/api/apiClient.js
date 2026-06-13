// API Client for 404 Found Backend
// Configure the backend URL - adjust based on your deployment

const API = {
  // Use the current domain if deployed together, otherwise specify backend URL
  baseURL: import.meta.env.VITE_API_URL || window.location.origin,

  getHeaders(extraHeaders = {}) {
    const headers = { ...extraHeaders };
    const token = localStorage.getItem("token");
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
  },

  async login(email, password) {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Invalid credentials");
      }
      
      const data = await response.json();
      if (data.token) {
        localStorage.setItem("token", data.token);
        localStorage.setItem("user", JSON.stringify(data.user));
      }
      return data;
    } catch (error) {
      console.error("Login API error:", error);
      throw error;
    }
  },

  async register(email, password, name, rollNumber = "N/A", role = "student") {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ email, password, name, roll_number: rollNumber, role })
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Registration failed");
      }
      
      const data = await response.json();
      if (data.token) {
        localStorage.setItem("token", data.token);
        localStorage.setItem("user", JSON.stringify(data.user));
      }
      return data;
    } catch (error) {
      console.error("Registration API error:", error);
      throw error;
    }
  },

  async getMe() {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/me`, {
        headers: this.getHeaders()
      });
      if (!response.ok) throw new Error("Session invalid");
      return await response.json();
    } catch (error) {
      console.error("GetMe API error:", error);
      throw error;
    }
  },

  async getUsers() {
    try {
      const response = await fetch(`${this.baseURL}/api/users`, {
        headers: this.getHeaders()
      });
      if (!response.ok) throw new Error("Failed to fetch users");
      return await response.json();
    } catch (error) {
      console.error("Error fetching users:", error);
      return { users: [] };
    }
  },


  logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  },

  getCurrentUser() {
    try {
      const userStr = localStorage.getItem("user");
      return userStr ? JSON.parse(userStr) : null;
    } catch (e) {
      return null;
    }
  },

  async getAllFound() {
    try {
      const response = await fetch(`${this.baseURL}/all-found`, {
        headers: this.getHeaders()
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error("Error fetching found items:", error);
      return { items: [], total: 0 };
    }
  },

  async reportFound(formData) {
    try {
      const response = await fetch(`${this.baseURL}/report-found`, {
        method: "POST",
        headers: this.getHeaders(), // Appends authorization token
        body: formData, // Already FormData with file
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error("Error reporting found item:", error);
      throw error;
    }
  },

  async searchLost(searchData) {
    try {
      const formData = new FormData();
      
      if (searchData.file) {
        formData.append("file", searchData.file);
      } else if (searchData.text_query) {
        formData.append("text_query", searchData.text_query);
      }

      const response = await fetch(`${this.baseURL}/search-lost`, {
        method: "POST",
        headers: this.getHeaders(), // Appends authorization token
        body: formData,
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error searching lost items:", error);
      throw error;
    }
  },

  async updateItemStatus(filename, status, claimedBy = null, claimedByName = null) {
    try {
      const response = await fetch(`${this.baseURL}/api/items/${filename}/status`, {
        method: "POST",
        headers: this.getHeaders({
          "Content-Type": "application/json"
        }),
        body: JSON.stringify({
          status,
          claimed_by: claimedBy,
          claimed_by_name: claimedByName
        })
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to update status");
      }
      return await response.json();
    } catch (error) {
      console.error("Error updating status:", error);
      throw error;
    }
  },

  async deleteItem(filename) {
    try {
      const response = await fetch(`${this.baseURL}/api/items/${filename}`, {
        method: "DELETE",
        headers: this.getHeaders()
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to delete item");
      }
      return await response.json();
    } catch (error) {
      console.error("Error deleting item:", error);
      throw error;
    }
  },

  // Helper to get image URL for found items
  getImageUrl(filename) {
    return `${this.baseURL}/found_items/${filename}`;
  },
};

export default API;
