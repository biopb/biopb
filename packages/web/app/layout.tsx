import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import { ClientBootstrap } from "./ClientBootstrap";

export const metadata: Metadata = {
  title: "BioPB Image Browser",
  description: "Website frontend for browsing and visualizing tensor-backed imaging data",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ClientBootstrap />
        {children}
      </body>
    </html>
  );
}
