import { Navigate, Link } from "react-router-dom";
import API from "../api/apiClient";

export default function ProtectedRoute({ children, allowedRoles }) {
  const user = API.getCurrentUser();
  const token = localStorage.getItem("token");

  if (!token || !user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return (
      <div className="page flex items-center justify-center" style={{ padding: "var(--space-4)" }}>
        <div className="bg-blob bg-blob-1"></div>
        <div className="bg-blob bg-blob-2"></div>
        <div className="card animate-scale-in" style={{ maxWidth: 420, width: "100%", textAlign: "center" }}>
          <div className="card-body" style={{ padding: "var(--space-10) var(--space-8)" }}>
            <div style={{
              width: 72,
              height: 72,
              borderRadius: "50%",
              background: "var(--danger-subtle)",
              color: "var(--danger)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "2rem",
              margin: "0 auto var(--space-6)",
              border: "1px solid rgba(239,68,68,0.15)",
            }}>
              <i className="fas fa-lock"></i>
            </div>
            <h2 style={{ fontSize: "1.5rem", marginBottom: "var(--space-3)" }}>Access Denied</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: "var(--space-6)", lineHeight: 1.6 }}>
              You do not have permission to view this page. Your current role is{" "}
              <span style={{ color: "var(--warning)", fontWeight: 700, textTransform: "capitalize" }}>{user.role}</span>.
            </p>
            <Link
              to={user.role === "guard" ? "/guard" : user.role === "admin" ? "/admin" : "/"}
              className="btn-primary btn-full btn-lg"
              style={{ textDecoration: "none" }}
            >
              <i className="fas fa-arrow-left"></i>Go to My Dashboard
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return children;
}
