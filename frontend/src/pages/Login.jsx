import { useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../api/apiClient";

export default function Login() {
  const navigate = useNavigate();
  const [isRegister, setIsRegister] = useState(false);
  const [role, setRole] = useState("student"); // student, guard, admin
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [rollNumber, setRollNumber] = useState("");
  const [loading, setLoading] = useState(false);
  
  // Custom Toast state
  const [toast, setToast] = useState({ message: "", type: "", visible: false });

  const showToast = (message, type = "error") => {
    setToast({ message, type, visible: true });
    setTimeout(() => {
      setToast((prev) => ({ ...prev, visible: false }));
    }, 4000);
  };

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
      showToast("Please fill in all required fields", "warning");
      return;
    }

    setLoading(true);
    try {
      if (isRegister) {
        const data = await API.register(email, password, name, rollNumber, role);
        showToast("Registration successful! Redirecting...", "success");
        setTimeout(() => {
          redirectUser(data.user.role);
        }, 1000);
      } else {
        const data = await API.login(email, password);
        showToast("Login successful! Redirecting...", "success");
        setTimeout(() => {
          redirectUser(data.user.role);
        }, 1000);
      }
    } catch (err) {
      showToast(err.message || "An error occurred during authentication", "error");
    } finally {
      setLoading(false);
    }
  };

  const redirectUser = (userRole) => {
    if (userRole === "admin") {
      navigate("/admin");
    } else if (userRole === "guard") {
      navigate("/guard");
    } else {
      navigate("/");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center relative overflow-hidden px-4">
      {/* Decorative background blobs */}
      <div className="absolute w-[400px] h-[400px] rounded-full bg-blue-600/10 blur-[80px] -top-20 -left-20 animate-pulse"></div>
      <div className="absolute w-[500px] h-[500px] rounded-full bg-purple-600/10 blur-[100px] -bottom-20 -right-20 animate-pulse duration-5000"></div>
      <div className="absolute w-[300px] h-[300px] rounded-full bg-emerald-600/5 blur-[70px] top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2"></div>

      {/* Brand Header */}
      <div className="z-10 mb-8 text-center flex flex-col items-center">
        <div className="bg-gradient-to-tr from-blue-500 to-indigo-600 text-white w-14 h-14 rounded-2xl flex items-center justify-center font-black shadow-lg shadow-indigo-500/20 mb-3 text-2xl border border-white/10 hover:rotate-12 transition duration-300">
          <i className="fas fa-search-plus"></i>
        </div>
        <h1 className="text-4xl font-extrabold text-white tracking-tight m-0 bg-clip-text text-transparent bg-gradient-to-r from-white via-indigo-100 to-indigo-300">
          404 Found
        </h1>
        <p className="text-indigo-400 text-sm font-medium mt-1">
          Intelligent Campus Lost & Found System
        </p>
      </div>

      {/* Main Glassmorphism Card */}
      <div className="z-10 w-full max-w-[460px] bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 p-8 rounded-3xl shadow-2xl shadow-black/50 transition-all duration-300">
        
        {/* Toggle Login/Register */}
        <div className="flex bg-slate-950/80 p-1.5 rounded-2xl mb-8 border border-slate-800/60">
          <button
            type="button"
            onClick={() => {
              setIsRegister(false);
              setEmail("");
              setPassword("");
              setName("");
              setRollNumber("");
            }}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
              !isRegister
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-indigo-600/10"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setIsRegister(true);
              setEmail("");
              setPassword("");
              setName("");
              setRollNumber("");
            }}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
              isRegister
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-indigo-600/10"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Create Account
          </button>
        </div>

        {/* Form Header */}
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">
            {isRegister ? "Start Your Journey" : "Welcome Back"}
          </h2>
          <p className="text-slate-400 text-sm">
            {isRegister ? "Sign up to register lost or found items" : "Access your campus dashboard"}
          </p>
        </div>

        {/* Role Selector Tabs (Only show register role selection or allow selecting for both) */}
        <div className="mb-6">
          <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-2.5">
            Select Role
          </label>
          <div className="grid grid-cols-3 gap-2">
            {[
              { id: "student", label: "Student", icon: "fa-user-graduate" },
              { id: "guard", label: "Guard", icon: "fa-user-shield" },
              { id: "admin", label: "Admin", icon: "fa-shield-halved" },
            ].map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => setRole(r.id)}
                className={`flex flex-col items-center py-2.5 px-1.5 rounded-xl border transition-all duration-300 ${
                  role === r.id
                    ? "bg-indigo-600/10 border-indigo-500 text-indigo-400 font-semibold"
                    : "bg-slate-950/40 border-slate-800 text-slate-500 hover:text-slate-300 hover:border-slate-700"
                }`}
              >
                <i className={`fas ${r.icon} mb-1.5 text-base`}></i>
                <span className="text-xs">{r.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Quick Demo Credentials Fill Button */}
        <div className="mb-6 p-3 bg-slate-950/60 border border-slate-800/40 rounded-2xl flex items-center justify-between">
          <div>
            <p className="text-xs font-bold text-slate-300">Developer Tools</p>
            <p className="text-[11px] text-slate-500">Auto-fill verified demo accounts</p>
          </div>
          <button
            type="button"
            onClick={() => handlePreFill(role)}
            className="bg-indigo-600/10 border border-indigo-500/30 hover:bg-indigo-600/20 text-indigo-400 px-3 py-1.5 rounded-xl text-xs font-semibold transition"
          >
            <i className="fas fa-magic mr-1"></i>Auto Fill
          </button>
        </div>

        {/* Actual Authentication Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          
          {/* Register Name field */}
          {isRegister && (
            <div>
              <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-1.5">
                Full Name
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 text-sm">
                  <i className="fas fa-user"></i>
                </span>
                <input
                  type="text"
                  placeholder="Enter your name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-slate-950/50 border border-slate-800 text-white rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-600 transition"
                  required
                />
              </div>
            </div>
          )}

          {/* Email field */}
          <div>
            <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 text-sm">
                <i className="fas fa-envelope"></i>
              </span>
              <input
                type="email"
                placeholder={role === "student" ? "username@iiitd.ac.in" : "staff@iiitd.ac.in"}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-slate-950/50 border border-slate-800 text-white rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-600 transition"
                required
              />
            </div>
          </div>

          {/* Roll Number field (Only for student registration) */}
          {isRegister && role === "student" && (
            <div>
              <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-1.5">
                Roll Number
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 text-sm">
                  <i className="fas fa-id-card"></i>
                </span>
                <input
                  type="text"
                  placeholder="E.g., 2023504"
                  value={rollNumber}
                  onChange={(e) => setRollNumber(e.target.value)}
                  className="w-full bg-slate-950/50 border border-slate-800 text-white rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-600 transition"
                  required
                />
              </div>
            </div>
          )}

          {/* Password field */}
          <div>
            <label className="block text-slate-400 text-xs font-bold uppercase tracking-wider mb-1.5">
              Password
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 text-sm">
                <i className="fas fa-lock"></i>
              </span>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-slate-950/50 border border-slate-800 text-white rounded-xl py-3 pl-10 pr-4 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 placeholder-slate-600 transition"
                required
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold py-3.5 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition duration-300 shadow-lg shadow-indigo-600/20 active:translate-y-0 disabled:opacity-50 flex items-center justify-center gap-2"
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

      {/* Floating custom Toast container */}
      {toast.visible && (
        <div
          className={`z-50 fixed bottom-6 right-6 flex items-center gap-3 py-3 px-5 rounded-2xl border shadow-2xl transition-all duration-300 translate-y-0 ${
            toast.type === "success"
              ? "bg-emerald-950/95 border-emerald-800 text-emerald-300"
              : toast.type === "warning"
              ? "bg-amber-950/95 border-amber-800 text-amber-300"
              : "bg-red-950/95 border-red-800 text-red-300"
          }`}
        >
          <div className="text-lg">
            <i
              className={`fas ${
                toast.type === "success"
                  ? "fa-circle-check"
                  : toast.type === "warning"
                  ? "fa-circle-exclamation"
                  : "fa-circle-xmark"
              }`}
            ></i>
          </div>
          <div className="font-semibold text-sm">{toast.message}</div>
        </div>
      )}
    </div>
  );
}
