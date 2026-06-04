import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Training } from './pages/Training';
import { Agents } from './pages/Agents';
import { Simulation } from './pages/Simulation';
import { Telemetry } from './pages/Telemetry';
import { Settings } from './pages/Settings';
import { useTelemetry } from './hooks/useTelemetry';

function App() {
  const { connected } = useTelemetry();

  return (
    <Router>
      <Layout connected={connected}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/telemetry" element={<Telemetry />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
