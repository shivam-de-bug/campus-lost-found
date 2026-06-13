import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import API from "../api/apiClient";

export default function Header() {
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);
  const user = API.getCurrentUser();

  const handleLogout = () => {
    API.logout();
    navigate("/login");
  };

  const getDashboardLink = () => {
    if (!user) return "/";
    if (user.role === "admin") return "/admin";
    if (user.role === "guard") return "/guard";
    return "/";
  };

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        {/* Brand logo */}
        <Link to={getDashboardLink()} className="flex items-center gap-2.5 no-underline group">
          <div className="bg-gradient-to-tr from-blue-600 to-indigo-600 text-white w-10 h-10 rounded-xl flex items-center justify-center font-black shadow-md shadow-indigo-600/10 group-hover:scale-105 transition duration-300">
            <i className="fas fa-search-plus text-base"></i>
          </div>
          <h1 className="text-xl font-black text-slate-800 m-0 group-hover:text-indigo-600 transition">
            404 Found
          </h1>
        </Link>

        {/* User context menu */}
        {user ? (
          <div className="relative">
            <div 
              onClick={() => setShowDropdown(!showDropdown)}
              className="flex items-center gap-3 cursor-pointer select-none bg-slate-50 border hover:bg-slate-100/80 px-3.5 py-2 rounded-2xl transition duration-300"
            >
              {/* User avatar / placeholder icon */}
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 text-white font-bold flex items-center justify-center shadow-sm">
                {user.name.charAt(0).toUpperCase()}
              </div>
              
              <div className="hidden md:block">
                <span className="font-bold text-slate-700 text-sm block leading-none">{user.name}</span>
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mt-1">
                  {user.role === "student" ? `Student • ${user.roll_number}` : user.role}
                </span>
              </div>
              
              <span className="text-slate-400 text-xs">
                <i className={`fas fa-chevron-${showDropdown ? "up" : "down"}`}></i>
              </span>
            </div>

            {/* Dropdown Menu */}
            {showDropdown && (
              <div className="absolute right-0 mt-2.5 w-60 bg-white border border-slate-200/80 rounded-2xl shadow-xl py-3 z-50 animate-in fade-in slide-in-from-top-3 duration-200">
                <div className="px-4 py-2 border-b border-slate-100">
                  <p className="font-bold text-slate-800 text-sm">{user.name}</p>
                  <p className="text-slate-400 text-xs truncate mt-0.5">{user.email}</p>
                </div>
                
                <div className="p-2 space-y-1">
                  <Link
                    to={getDashboardLink()}
                    onClick={() => setShowDropdown(false)}
                    className="w-full flex items-center gap-3 px-3 py-2 text-xs font-semibold text-slate-600 hover:text-indigo-600 hover:bg-indigo-50/50 rounded-xl transition no-underline"
                  >
                    <i className="fas fa-gauge w-4"></i>My Dashboard
                  </Link>

                  {user.role === "admin" && (
                    <Link
                      to="/"
                      onClick={() => setShowDropdown(false)}
                      className="w-full flex items-center gap-3 px-3 py-2 text-xs font-semibold text-slate-600 hover:text-indigo-600 hover:bg-indigo-50/50 rounded-xl transition no-underline"
                    >
                      <i className="fas fa-home w-4"></i>Student Panel View
                    </Link>
                  )}

                  <button
                    onClick={() => {
                      setShowDropdown(false);
                      handleLogout();
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 text-xs font-bold text-rose-600 hover:bg-rose-50 rounded-xl transition text-left"
                  >
                    <i className="fas fa-sign-out-alt w-4"></i>Sign Out
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <Link
            to="/login"
            className="bg-indigo-600 text-white px-5 py-2.5 rounded-xl hover:bg-indigo-700 transition font-bold text-sm shadow-md shadow-indigo-600/10 no-underline"
          >
            Access Platform
          </Link>
        )}
      </div>
    </header>
  );
}
