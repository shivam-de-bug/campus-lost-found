import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ToastProvider } from "./components/ToastContext";
import Toast from "./components/Toast";
import Home from "./pages/Home";
import Admin from "./pages/Admin";
import Guard from "./pages/Guard";
import Login from "./pages/Login";
import ProtectedRoute from "./components/ProtectedRoute";
import "./App.css";

function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          {/* Public login route */}
          <Route path="/login" element={<Login />} />

          {/* User / Student Home route */}
          <Route
            path="/"
            element={
              <ProtectedRoute allowedRoles={["student", "guard", "admin"]}>
                <Home />
              </ProtectedRoute>
            }
          />

          {/* Security Guard inventory management route */}
          <Route
            path="/guard"
            element={
              <ProtectedRoute allowedRoles={["guard", "admin"]}>
                <Guard />
              </ProtectedRoute>
            }
          />

          {/* Platform Control Panel / Admin route */}
          <Route
            path="/admin"
            element={
              <ProtectedRoute allowedRoles={["admin"]}>
                <Admin />
              </ProtectedRoute>
            }
          />

          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
      <Toast />
    </ToastProvider>
  );
}

export default App;
