import { Outlet } from 'react-router-dom'
import Header from './Header'

export default function Layout() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-f1-black via-f1-gray-900 to-f1-black">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <Outlet />
      </main>
      <footer className="container mx-auto px-4 py-8 mt-16 border-t border-f1-gray-800">
        <div className="text-center text-f1-gray-400 text-sm">
          <p>© 2025 F1 Predictions. Powered by AI and FastF1 data.</p>
          <p className="mt-2">
            Created by{' '}
            <a
              href="https://instagram.com/mar_antaya"
              target="_blank"
              rel="noopener noreferrer"
              className="text-f1-red hover:underline"
            >
              @mar_antaya
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}