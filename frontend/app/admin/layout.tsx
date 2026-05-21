import AdminGuard from '@/components/AdminGuard'
import Link from 'next/link'
import { LayoutDashboard, FileText, Activity, ArrowLeft } from 'lucide-react'

export const metadata = { title: 'Admin · KisanCredit' }

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <AdminGuard>
      <div className="min-h-screen bg-stone-50">
        <header className="bg-white border-b border-stone-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-6">
              <span className="font-bold text-stone-900">KisanCredit · Operator</span>
              <nav className="hidden sm:flex items-center gap-4 text-sm">
                <NavLink href="/admin" icon={<LayoutDashboard className="w-4 h-4" />}>Overview</NavLink>
                <NavLink href="/admin/applications" icon={<FileText className="w-4 h-4" />}>Applications</NavLink>
                <NavLink href="/admin/drift" icon={<Activity className="w-4 h-4" />}>Drift</NavLink>
              </nav>
            </div>
            <Link href="/" className="text-sm text-stone-600 hover:text-stone-900 inline-flex items-center gap-1">
              <ArrowLeft className="w-3 h-3" /> Back to site
            </Link>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">{children}</main>
      </div>
    </AdminGuard>
  )
}

function NavLink({ href, icon, children }: { href: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <Link href={href} className="text-stone-600 hover:text-stone-900 inline-flex items-center gap-1.5">
      {icon} {children}
    </Link>
  )
}
