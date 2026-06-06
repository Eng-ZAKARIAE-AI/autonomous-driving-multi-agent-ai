import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Training } from './pages/Training';
import { Agents } from './pages/Agents';
import { Simulation } from './pages/Simulation';
import { Telemetry } from './pages/Telemetry';
import { Settings } from './pages/Settings';
import { TelemetryProvider, useTelemetryContext } from './context/TelemetryContext';

function AppShell() {
  const { connected } = useTelemetryContext();

  return (
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
  );
}

function App() {
  return (
    <Router>
      <TelemetryProvider>
        <AppShell />
      </TelemetryProvider>
    </Router>
  );
}

export default App;
