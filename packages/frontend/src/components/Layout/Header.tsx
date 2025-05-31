import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Trophy, TrendingUp, Calendar, Users, BarChart3, Menu, X } from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

const navigation = [
  { name: 'Predictions', href: '/', icon: Trophy },
  { name: 'Calendar', href: '/calendar', icon: Calendar },
  { name: 'Drivers', href: '/drivers', icon: Users },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Historical', href: '/historical', icon: TrendingUp },
]

export default function Header() {
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 glass-effect border-b border-f1-gray-800">
      <nav className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-f1-red rounded-lg flex items-center justify-center">
              <Trophy className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">F1 Predictions</h1>
              <p className="text-xs text-f1-gray-400">2025 Season</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <ul className="hidden md:flex items-center space-x-8">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={clsx(
                      'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                      isActive
                        ? 'bg-f1-red text-white'
                        : 'text-f1-gray-300 hover:text-white hover:bg-f1-gray-800'
                    )}
                  >
                    <item.icon className="w-4 h-4" />
                    <span className="font-medium">{item.name}</span>
                  </Link>
                </li>
              )
            })}
          </ul>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 rounded-lg hover:bg-f1-gray-800 transition-colors"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden mt-4 py-4 border-t border-f1-gray-800"
          >
            <ul className="space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className={clsx(
                        'flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200',
                        isActive
                          ? 'bg-f1-red text-white'
                          : 'text-f1-gray-300 hover:text-white hover:bg-f1-gray-800'
                      )}
                    >
                      <item.icon className="w-5 h-5" />
                      <span className="font-medium">{item.name}</span>
                    </Link>
                  </li>
                )
              })}
            </ul>
          </motion.div>
        )}
      </nav>
    </header>
  )
}