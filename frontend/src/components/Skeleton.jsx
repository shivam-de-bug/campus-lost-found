export function SkeletonCard() {
  return (
    <div className="card" style={{ pointerEvents: "none" }}>
      <div className="skeleton" style={{ height: 200, borderRadius: "var(--radius-2xl) var(--radius-2xl) 0 0" }}></div>
      <div style={{ padding: "var(--space-5)" }}>
        <div className="skeleton skeleton-text" style={{ width: "70%", marginBottom: 12 }}></div>
        <div className="skeleton skeleton-text-sm" style={{ width: "90%", marginBottom: 8 }}></div>
        <div className="skeleton skeleton-text-sm" style={{ width: "60%", marginBottom: 8 }}></div>
        <div className="skeleton skeleton-text-sm" style={{ width: "75%" }}></div>
      </div>
    </div>
  );
}

export function SkeletonRow() {
  return (
    <tr>
      <td><div className="skeleton" style={{ width: 48, height: 48, borderRadius: "var(--radius-md)" }}></div></td>
      <td><div className="skeleton skeleton-text" style={{ width: "80%" }}></div></td>
      <td><div className="skeleton skeleton-text" style={{ width: "60%" }}></div></td>
      <td><div className="skeleton skeleton-text" style={{ width: "50%" }}></div></td>
    </tr>
  );
}

export function SkeletonGrid({ count = 4 }) {
  return (
    <div className="grid grid-4">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}
