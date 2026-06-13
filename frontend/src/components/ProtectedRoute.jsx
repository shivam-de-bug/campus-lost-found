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
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-indigo-950 flex flex-col justify-center items-center text-white px-4">
        <div className="bg-white/10 backdrop-blur-md border border-white/25 p-8 rounded-2xl shadow-2xl max-w-md w-full text-center">
          <div className="w-20 h-20 bg-red-500/20 text-red-400 rounded-full flex items-center justify-center text-4xl mx-auto mb-6 border border-red-500/30">
            <i className="fas fa-exclamation-triangle"></i>
          </div>
          <h2 className="text-3xl font-extrabold mb-3 tracking-tight text-white">Access Denied</h2>
          <p className="text-indigo-200 mb-6 leading-relaxed">
            You do not have permission to view this dashboard. Your current role is <span className="font-bold text-amber-400 capitalize">{user.role}</span>.
          </p>
          <Link
            to={user.role === "guard" ? "/guard" : user.role === "admin" ? "/admin" : "/"}
            className="w-full block bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold py-3 px-6 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition duration-300 shadow-lg"
          >
            <i className="fas fa-arrow-left mr-2"></i>Go to My Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return children;
}
