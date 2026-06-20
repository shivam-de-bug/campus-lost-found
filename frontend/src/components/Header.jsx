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
    <header className="app-header">
      <div className="app-header-inner">
        {/* Brand logo */}
        <Link to={getDashboardLink()} className="flex items-center gap-3 no-underline" style={{ textDecoration: "none" }}>
          <div style={{
            background: "linear-gradient(135deg, var(--primary), #a855f7)",
            color: "white",
            width: 38,
            height: 38,
            borderRadius: "var(--radius-lg)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1rem",
            boxShadow: "0 4px 12px var(--primary-glow)",
            transition: "transform var(--transition-base)",
          }}>
            <i className="fas fa-search-plus"></i>
          </div>
          <h1 style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: "1.2rem",
            fontWeight: 800,
            margin: 0,
            background: "linear-gradient(135deg, var(--text-primary), var(--primary-light))",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            404 Found
          </h1>
        </Link>

        {/* User context menu */}
        {user ? (
          <div className="relative">
            <div
              onClick={() => setShowDropdown(!showDropdown)}
              className="flex items-center gap-3 cursor-pointer select-none"
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                padding: "6px 12px",
                borderRadius: "var(--radius-xl)",
                transition: "all var(--transition-fast)",
              }}
            >
              {/* User avatar */}
              <div style={{
                width: 32,
                height: 32,
                borderRadius: "var(--radius-md)",
                background: "linear-gradient(135deg, var(--primary), #a855f7)",
                color: "white",
                fontWeight: 700,
                fontSize: "0.85rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                {user.name.charAt(0).toUpperCase()}
              </div>

              <div className="sm-hidden">
                <span style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.85rem", display: "block", lineHeight: 1.2 }}>
                  {user.name}
                </span>
                <span style={{ fontSize: "0.6rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginTop: 2 }}>
                  {user.role === "student" ? `Student • ${user.roll_number}` : user.role}
                </span>
              </div>

              <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                <i className={`fas fa-chevron-${showDropdown ? "up" : "down"}`}></i>
              </span>
            </div>

            {/* Dropdown Menu */}
            {showDropdown && (
              <div className="dropdown">
                <div style={{ padding: "var(--space-3) var(--space-4)", borderBottom: "1px solid var(--border)" }}>
                  <p style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.85rem" }}>{user.name}</p>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: 2 }}>{user.email}</p>
                </div>

                <div style={{ padding: "var(--space-1)" }}>
                  <Link
                    to={getDashboardLink()}
                    onClick={() => setShowDropdown(false)}
                    className="dropdown-item"
                  >
                    <i className="fas fa-gauge" style={{ width: 16 }}></i>My Dashboard
                  </Link>

                  {user.role === "admin" && (
                    <Link
                      to="/"
                      onClick={() => setShowDropdown(false)}
                      className="dropdown-item"
                    >
                      <i className="fas fa-home" style={{ width: 16 }}></i>Student Panel View
                    </Link>
                  )}

                  <div className="dropdown-divider"></div>

                  <button
                    onClick={() => {
                      setShowDropdown(false);
                      handleLogout();
                    }}
                    className="dropdown-item dropdown-item-danger"
                  >
                    <i className="fas fa-sign-out-alt" style={{ width: 16 }}></i>Sign Out
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <Link
            to="/login"
            className="btn-primary"
            style={{ textDecoration: "none", fontSize: "0.85rem", padding: "8px 20px" }}
          >
            Access Platform
          </Link>
        )}
      </div>
    </header>
  );
}
