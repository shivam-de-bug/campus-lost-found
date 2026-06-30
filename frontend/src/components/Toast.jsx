import { useToast } from "./ToastContext";

const ICON_MAP = {
  success: "fa-circle-check",
  error: "fa-circle-xmark",
  warning: "fa-circle-exclamation",
  info: "fa-circle-info",
};

export default function Toast() {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.type}`}>
          <span className="toast-icon">
            <i className={`fas ${ICON_MAP[toast.type] || ICON_MAP.info}`}></i>
          </span>
          <span className="toast-message">{toast.message}</span>
          <button className="toast-close" onClick={() => removeToast(toast.id)}>
            <i className="fas fa-times"></i>
          </button>
        </div>
      ))}
    </div>
  );
}
