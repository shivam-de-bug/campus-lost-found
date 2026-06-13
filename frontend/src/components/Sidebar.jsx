import { Link } from "react-router-dom";

export default function Sidebar({ activeSection, onSectionChange }) {
  const sections = [
    { id: "dashboard", icon: "fa-chart-pie", title: "Dashboard" },
    { id: "items", icon: "fa-boxes-stacked", title: "Items" },
    { id: "disputes", icon: "fa-gavel", title: "Disputes" },
    { id: "analytics", icon: "fa-chart-line", title: "Trends" },
    { id: "reports", icon: "fa-file-invoice", title: "Reports" },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-lg border-t border-slate-200/80 shadow-[0_-4px_20px_rgba(0,0,0,0.03)] z-40 flex justify-around items-center h-16 px-4 pb-safe-bottom">
      {/* Home link */}
      <Link
        to="/"
        className="flex flex-col items-center justify-center text-slate-400 hover:text-slate-600 transition duration-300 no-underline"
        title="Student View"
      >
        <span className="w-5 h-5 flex items-center justify-center text-base">
          <i className="fas fa-home"></i>
        </span>
        <span className="text-[9px] font-bold mt-1 uppercase tracking-wider">Home</span>
      </Link>

      {/* Admin Sections */}
      {sections.map((section) => {
        const isActive = activeSection === section.id;
        return (
          <button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            className="flex flex-col items-center justify-center relative bg-transparent border-0 outline-none cursor-pointer flex-1 py-1"
          >
            {isActive && (
              <span className="absolute -top-1 w-6 h-1 bg-indigo-600 rounded-full animate-pulse"></span>
            )}
            <span className={`w-6 h-6 flex items-center justify-center text-base transition-all duration-300 ${
              isActive ? "text-indigo-600 scale-110" : "text-slate-400 hover:text-slate-600"
            }`}>
              <i className={`fas ${section.icon}`}></i>
            </span>
            <span className={`text-[9px] font-bold mt-1 uppercase tracking-wider transition-colors duration-300 ${
              isActive ? "text-indigo-600 font-extrabold" : "text-slate-400"
            }`}>
              {section.title}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
