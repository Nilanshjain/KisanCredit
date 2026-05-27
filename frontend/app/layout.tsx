import type { Metadata } from "next"
import "./globals.css"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import Navbar from "@/components/Navbar"

export const metadata: Metadata = {
  title: "KisanCredit — ML credit scoring for thin-file borrowers",
  description: "LightGBM default-risk model trained on 307K real loan applicants, served through a production stack with a lender operations console, SHAP explanations, and Gemini-generated reasoning.",
  keywords: ["credit scoring", "LightGBM", "SHAP", "Home Credit", "machine learning", "FastAPI", "Next.js"],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" style={{ colorScheme: 'light' }}>
      <body style={{ color: '#1C1917', backgroundColor: '#FAFAF9' }}>
        <ErrorBoundary>
          <Navbar />
          {/* Navbar is fixed at h-14 (56px); pt-14 keeps page content clear of it. */}
          <main className="pt-14">{children}</main>
        </ErrorBoundary>
      </body>
    </html>
  )
}
