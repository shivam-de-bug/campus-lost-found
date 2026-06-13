// API Client for 404 Found Backend
// Configure the backend URL - adjust based on your deployment

const API = {
  // Use the current domain if deployed together, otherwise specify backend URL
  baseURL: import.meta.env.VITE_API_URL || window.location.origin,

  async getAllFound() {
    try {
      const response = await fetch(`${this.baseURL}/all-found`);
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
        body: formData,
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error("Error searching lost items:", error);
      throw error;
    }
  },

  // Helper to get image URL for found items
  getImageUrl(filename) {
    return `${this.baseURL}/found_items/${filename}`;
  },
};

export default API;
