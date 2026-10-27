"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  ["Overview", "/"],
  ["AI systems", "/systems"],
  ["Test suites", "/suites"],
  ["Test plans", "/plans"],
  ["Test runs", "/runs"],
  ["Findings", "/findings"],
  ["Developers", "/developers"],
  ["Resources", "/resources"]
] as const;

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">GK</span>
        <span>
          <strong>Gatekeeper</strong>
          <small>Healthcare AI assurance</small>
        </span>
      </div>
      <nav aria-label="Primary">
        {navigation.map(([label, href]) => {
          const active = href === "/" ? pathname === href : pathname.startsWith(href);
          return (
            <Link className={active ? "nav-link active" : "nav-link"} href={href} key={href}>
              <span className="nav-dot" aria-hidden="true" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="sidebar-note">
        <span className="eyebrow">Assurance principle</span>
        <p>Technical failure never becomes PASS.</p>
      </div>
    </aside>
  );
}
