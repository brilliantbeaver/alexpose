import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { TopNavBar } from "@/components/navigation/TopNavBar";
import { Breadcrumbs } from "@/components/navigation/Breadcrumbs";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AlexPose - Gait Analysis System",
  description: "AI-powered gait analysis for health condition identification",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <TopNavBar />
        <div className="container mx-auto px-4 py-6">
          <Breadcrumbs />
          <main className="mt-4">{children}</main>
        </div>
      </body>
    </html>
  );
}
