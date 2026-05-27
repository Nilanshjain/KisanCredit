import AdminGuard from '@/components/AdminGuard'
import AdminTabs from './AdminTabs'

export const metadata = { title: 'Operator console · KisanCredit' }

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <AdminGuard>
      <div className="min-h-screen bg-stone-50">
        <div className="border-b hairline bg-white">
          <div className="max-w-6xl mx-auto px-6">
            <div className="pt-8 pb-1">
              <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">
                Operator console
              </p>
            </div>
            <AdminTabs />
          </div>
        </div>
        <main className="max-w-6xl mx-auto px-6 py-10">{children}</main>
      </div>
    </AdminGuard>
  )
}
