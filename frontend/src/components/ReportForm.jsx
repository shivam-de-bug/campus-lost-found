import { useState } from "react";

export default function ReportForm({ onSubmit, loading, onCancel }) {
  const [formData, setFormData] = useState({
    description: "",
    location: "",
    contact: "",
    category: "",
  });
  const [reportFile, setReportFile] = useState(null);

  const categories = [
    "Electronics",
    "Clothing",
    "Accessories",
    "Documents",
    "Bags",
    "Footwear",
    "Other",
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await onSubmit(formData, reportFile);
    setFormData({
      description: "",
      location: "",
      contact: "",
      category: "",
    });
    setReportFile(null);
  };

  return (
    <div className="bg-white rounded-3xl border border-slate-200/80 p-8 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <h4 className="text-xl font-bold text-slate-800 m-0">Log Found Item</h4>
        {onCancel && (
          <button
            onClick={onCancel}
            className="text-slate-400 hover:text-slate-600 bg-slate-50 hover:bg-slate-100 p-2 rounded-full transition w-8 h-8 flex items-center justify-center"
          >
            <i className="fas fa-times text-xs"></i>
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Image Upload Zone */}
        <div>
          <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
            Upload Item Photo <span className="text-rose-500">*</span>
          </label>
          <label className="block cursor-pointer">
            <div className="border-2 border-dashed border-slate-200 hover:border-indigo-500 hover:bg-slate-50/50 rounded-2xl p-6 text-center transition duration-300 group">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setReportFile(e.target.files?.[0] || null)}
                required
                className="hidden"
              />
              <span className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-500 flex items-center justify-center text-lg mx-auto mb-2.5 group-hover:scale-110 transition duration-300">
                <i className="fas fa-camera"></i>
              </span>
              {reportFile ? (
                <p className="text-indigo-600 text-xs font-bold">{reportFile.name}</p>
              ) : (
                <>
                  <p className="text-slate-700 text-xs font-bold">Choose an image of the item</p>
                  <p className="text-[10px] text-slate-400 mt-1">Image matches best when item is centered</p>
                </>
              )}
            </div>
          </label>
        </div>

        {/* Description */}
        <div>
          <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
            Item Description
          </label>
          <div className="relative">
            <span className="absolute left-3.5 top-3.5 text-slate-400 text-sm">
              <i className="fas fa-pen"></i>
            </span>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleChange}
              placeholder="Describe the item's key features (e.g., color, brand, distinct marks)..."
              className="w-full bg-slate-50/50 border border-slate-200 text-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-400 transition"
              rows="3"
            />
          </div>
        </div>

        {/* Category */}
        <div>
          <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
            Category <span className="text-rose-500">*</span>
          </label>
          <div className="relative">
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 text-sm">
              <i className="fas fa-tag"></i>
            </span>
            <select
              name="category"
              value={formData.category}
              onChange={handleChange}
              className="w-full bg-slate-50/50 border border-slate-200 text-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition appearance-none cursor-pointer"
              required
            >
              <option value="">Select a category</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
            <span className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none text-xs">
              <i className="fas fa-chevron-down"></i>
            </span>
          </div>
        </div>

        {/* Location */}
        <div>
          <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
            Location Found <span className="text-rose-500">*</span>
          </label>
          <div className="relative">
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 text-sm">
              <i className="fas fa-map-marker-alt"></i>
            </span>
            <input
              type="text"
              name="location"
              value={formData.location}
              onChange={handleChange}
              placeholder="E.g., Library 2nd floor, Student Center tables..."
              className="w-full bg-slate-50/50 border border-slate-200 text-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-400 transition"
              required
            />
          </div>
        </div>

        {/* Contact */}
        <div>
          <label className="block text-slate-600 text-xs font-bold uppercase tracking-wider mb-2">
            Contact Details <span className="text-rose-500">*</span>
          </label>
          <div className="relative">
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 text-sm">
              <i className="fas fa-envelope"></i>
            </span>
            <input
              type="text"
              name="contact"
              value={formData.contact}
              onChange={handleChange}
              placeholder="E.g., phone number or email address..."
              className="w-full bg-slate-50/50 border border-slate-200 text-slate-800 rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-400 transition"
              required
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 pt-3">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-bold py-3 rounded-xl hover:from-emerald-700 hover:to-teal-700 transition duration-300 shadow-md shadow-emerald-600/10 disabled:opacity-50 flex items-center justify-center gap-2"
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
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold py-3 rounded-xl transition duration-300 text-center"
            >
              Cancel
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
