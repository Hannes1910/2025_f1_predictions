/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'f1-red': '#E10600',
        'f1-black': '#15151E',
        'f1-white': '#F7F4F1',
        'f1-gray': {
          100: '#F6F6F8',
          200: '#E8E8ED',
          300: '#D5D5DC',
          400: '#A8A8B5',
          500: '#787885',
          600: '#4A4A57',
          700: '#2F2F3A',
          800: '#1F1F28',
          900: '#15151E',
        },
        'team': {
          'redbull': '#0600EF',
          'mclaren': '#FF8700',
          'ferrari': '#DC0000',
          'mercedes': '#00D2BE',
          'astonmartin': '#006F62',
          'alpine': '#0090FF',
          'williams': '#005AFF',
          'racingbulls': '#2B4562',
          'kicksauber': '#00E701',
          'haas': '#FFFFFF',
        }
      },
      fontFamily: {
        'formula': ['Formula1-Display', 'sans-serif'],
        'titillium': ['Titillium Web', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}