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
    <nav className="bottom-nav">
      {/* Home link */}
      <Link to="/" className="nav-item" style={{ textDecoration: "none" }} title="Student View">
        <i className="fas fa-home"></i>
        <span>Home</span>
      </Link>

      {/* Admin Sections */}
      {sections.map((section) => (
        <button
          key={section.id}
          onClick={() => onSectionChange(section.id)}
          className={`nav-item ${activeSection === section.id ? "active" : ""}`}
        >
          <i className={`fas ${section.icon}`}></i>
          <span>{section.title}</span>
        </button>
      ))}
    </nav>
  );
}
