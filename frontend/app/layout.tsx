import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Sidebar } from "@/components/sidebar";
import { serverApiConfigurationPresent } from "@/lib/api";

import "./globals.css";

// API credentials and connectivity are runtime deployment state. Never bake
// authenticated data (or an unconfigured empty state) into the build output.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: { default: "Gatekeeper", template: "%s · Gatekeeper" },
  description: "Healthcare AI assurance and release gating"
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  const connected = serverApiConfigurationPresent();
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">
            <header className="topbar">
              <div>
                <span className={`connection-dot ${connected ? "connected" : ""}`} />
                {connected ? "API connected" : "API not configured"}
              </div>
              <div className="environment">PHI-safe operational view</div>
            </header>
            <div className="content">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
