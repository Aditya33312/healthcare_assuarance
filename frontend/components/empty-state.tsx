import Link from "next/link";

export function EmptyState({
  title,
  detail,
  action,
  href
}: {
  title: string;
  detail: string;
  action?: string;
  href?: string;
}) {
  return (
    <div className="empty-state">
      <div className="empty-symbol" aria-hidden="true">◇</div>
      <h3>{title}</h3>
      <p>{detail}</p>
      {action && href ? <Link className="button secondary" href={href}>{action}</Link> : null}
    </div>
  );
}

