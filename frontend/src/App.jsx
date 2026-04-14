import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Overview from './pages/Overview';
import Trends from './pages/Trends';
import { ThemeProvider } from './hooks/useTheme';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="app">
          <Header />
          <main>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/trends" element={<Trends />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
