import { useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../api/apiClient";
import { useToast } from "../components/ToastContext";

export default function Login() {
  const navigate = useNavigate();
  const toast = useToast();
  const [isRegister, setIsRegister] = useState(false);
  const [role, setRole] = useState("student");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [rollNumber, setRollNumber] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePreFill = (selectedRole) => {
    setRole(selectedRole);
    if (selectedRole === "student") {
      setEmail("student@iiitd.ac.in");
      setPassword("student123");
      setName("Demo Student");
      setRollNumber("2023504");
    } else if (selectedRole === "guard") {
      setEmail("guard@iiitd.ac.in");
      setPassword("guard123");
      setName("Demo Guard");
      setRollNumber("N/A");
    } else {
      setEmail("admin@iiitd.ac.in");
      setPassword("admin123");
      setName("Demo Admin");
      setRollNumber("N/A");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password || (isRegister && (!name || (role === "student" && !rollNumber)))) {
      toast.warning("Please fill in all required fields");
      return;
    }

    setLoading(true);
    try {
      if (isRegister) {
        const data = await API.register(email, password, name, rollNumber, role);
        toast.success("Registration successful! Redirecting...");
        setTimeout(() => redirectUser(data.user.role), 1000);
      } else {
        const data = await API.login(email, password);
        toast.success("Login successful! Redirecting...");
        setTimeout(() => redirectUser(data.user.role), 1000);
      }
    } catch (err) {
      toast.error(err.message || "An error occurred during authentication");
    } finally {
      setLoading(false);
    }
  };

  const redirectUser = (userRole) => {
    if (userRole === "admin") navigate("/admin");
    else if (userRole === "guard") navigate("/guard");
    else navigate("/");
  };

  return (
    <div className="page flex flex-col justify-center items-center relative overflow-hidden" style={{ padding: "var(--space-4)" }}>
      {/* Animated background blobs */}
      <div className="bg-blob bg-blob-1"></div>
      <div className="bg-blob bg-blob-2"></div>
      <div className="bg-blob bg-blob-3"></div>

      {/* Brand Header */}
      <div className="animate-slide-up" style={{ zIndex: 10, marginBottom: "var(--space-8)", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{
          background: "linear-gradient(135deg, var(--primary), #a855f7)",
          color: "white",
          width: 56,
          height: 56,
          borderRadius: "var(--radius-xl)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "1.5rem",
          boxShadow: "var(--shadow-glow-lg)",
          marginBottom: "var(--space-4)",
          border: "1px solid rgba(255,255,255,0.1)",
          transition: "transform var(--transition-base)",
          cursor: "pointer",
        }}>
          <i className="fas fa-search-plus"></i>
        </div>
        <h1 className="gradient-text" style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: "2.5rem",
          fontWeight: 900,
          margin: 0,
          letterSpacing: "-0.02em",
        }}>
          404 Found
        </h1>
        <p style={{ color: "var(--primary-light)", fontSize: "0.85rem", fontWeight: 500, marginTop: "var(--space-1)" }}>
          Intelligent Campus Lost & Found System
        </p>
      </div>

      {/* Main Card */}
      <div className="glass animate-scale-in" style={{
        zIndex: 10,
        width: "100%",
        maxWidth: 460,
        padding: "var(--space-8)",
        borderRadius: "var(--radius-2xl)",
        boxShadow: "var(--shadow-xl)",
      }}>

        {/* Toggle Login/Register */}
        <div className="tab-switcher mb-6">
          <button
            type="button"
            onClick={() => { setIsRegister(false); setEmail(""); setPassword(""); setName(""); setRollNumber(""); }}
            className={!isRegister ? "active" : ""}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => { setIsRegister(true); setEmail(""); setPassword(""); setName(""); setRollNumber(""); }}
            className={isRegister ? "active" : ""}
          >
            Create Account
          </button>
        </div>

        {/* Form Header */}
        <div style={{ textAlign: "center", marginBottom: "var(--space-6)" }}>
          <h2 style={{ fontSize: "1.4rem", marginBottom: "var(--space-2)" }}>
            {isRegister ? "Start Your Journey" : "Welcome Back"}
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            {isRegister ? "Sign up to register lost or found items" : "Access your campus dashboard"}
          </p>
        </div>

        {/* Role Selector */}
        <div className="mb-5">
          <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 10 }}>
            Select Role
          </label>
          <div className="role-grid">
            {[
              { id: "student", label: "Student", icon: "fa-user-graduate" },
              { id: "guard", label: "Guard", icon: "fa-user-shield" },
              { id: "admin", label: "Admin", icon: "fa-shield-halved" },
            ].map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => setRole(r.id)}
                className={`role-card ${role === r.id ? "active" : ""}`}
              >
                <i className={`fas ${r.icon}`}></i>
                <span>{r.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Quick Demo Credentials */}
        <div className="dev-banner mb-5">
          <div>
            <p style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-primary)" }}>Developer Tools</p>
            <p style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Auto-fill verified demo accounts</p>
          </div>
          <button
            type="button"
            onClick={() => handlePreFill(role)}
            style={{
              background: "var(--primary-subtle)",
              border: "1px solid rgba(124,58,237,0.2)",
              color: "var(--primary-light)",
              padding: "6px 12px",
              borderRadius: "var(--radius-lg)",
              fontSize: "0.75rem",
              fontWeight: 600,
            }}
          >
            <i className="fas fa-magic mr-1"></i>Auto Fill
          </button>
        </div>

        {/* Auth Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {isRegister && (
            <div>
              <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                Full Name
              </label>
              <div className="input-icon-wrapper">
                <span className="input-icon"><i className="fas fa-user"></i></span>
                <input type="text" placeholder="Enter your name" value={name} onChange={(e) => setName(e.target.value)} required />
              </div>
            </div>
          )}

          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
              Email Address
            </label>
            <div className="input-icon-wrapper">
              <span className="input-icon"><i className="fas fa-envelope"></i></span>
              <input
                type="email"
                placeholder={role === "student" ? "username@iiitd.ac.in" : "staff@iiitd.ac.in"}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          {isRegister && role === "student" && (
            <div>
              <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
                Roll Number
              </label>
              <div className="input-icon-wrapper">
                <span className="input-icon"><i className="fas fa-id-card"></i></span>
                <input type="text" placeholder="E.g., 2023504" value={rollNumber} onChange={(e) => setRollNumber(e.target.value)} required />
              </div>
            </div>
          )}

          <div>
            <label style={{ display: "block", fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: 6 }}>
              Password
            </label>
            <div className="input-icon-wrapper">
              <span className="input-icon"><i className="fas fa-lock"></i></span>
              <input type="password" placeholder="••••••••" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary btn-full btn-lg"
            style={{ marginTop: "var(--space-5)" }}
          >
            {loading ? (
              <>
                <i className="fas fa-circle-notch fa-spin"></i>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <i className={`fas ${isRegister ? "fa-user-plus" : "fa-sign-in-alt"}`}></i>
                <span>{isRegister ? "Create Account" : "Access Platform"}</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
