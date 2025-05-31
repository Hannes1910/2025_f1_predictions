import { Routes, Route } from 'react-router-dom'
import Layout from '@/components/Layout'
import Predictions from '@/pages/Predictions'
import Calendar from '@/pages/Calendar'
import Drivers from '@/pages/Drivers'
import Analytics from '@/pages/Analytics'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Predictions />} />
        <Route path="calendar" element={<Calendar />} />
        <Route path="drivers" element={<Drivers />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="historical" element={<div>Historical page coming soon...</div>} />
        <Route path="race/:raceId" element={<div>Race details page coming soon...</div>} />
      </Route>
    </Routes>
  )
}

export default App