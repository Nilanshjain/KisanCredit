import type { Metadata } from "next"
import "./globals.css"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import Navbar from "@/components/Navbar"

export const metadata: Metadata = {
  title: "KisanCredit - AI Credit Scoring for Rural India",
  description: "Alternative credit scoring platform using AI and machine learning. Get loan decisions in 60 seconds using SMS, UPI, and contact data—no traditional credit history required.",
  keywords: ["AI credit scoring", "alternative credit", "rural India", "farmer loans", "UPI", "SMS", "machine learning", "instant approval"],
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
          {children}
        </ErrorBoundary>
      </body>
    </html>
  )
}
