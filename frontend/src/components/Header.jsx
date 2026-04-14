import { useTheme } from '../hooks/useTheme';
import { NavLink } from 'react-router-dom';

export default function Header() {
    const { theme, toggleTheme } = useTheme();

    return (
        <header className="header">
            <div className="header-logo">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
                </svg>
                Panchayat
                <div className="header-logo-dot" title="Live Data Active"></div>
            </div>

            <nav className="header-nav">
                <NavLink
                    to="/"
                    end
                    className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                >
                    Overview
                </NavLink>
                <NavLink
                    to="/trends"
                    className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
                >
                    Trends &amp; Topics
                </NavLink>
            </nav>

            <div className="header-right">
                <div className="live-pill">
                    <span className="live-dot"></span>
                    Live
                </div>
                <button
                    className="theme-toggle"
                    onClick={toggleTheme}
                    aria-label="Toggle theme"
                    title="Toggle light/dark mode"
                >
                    {theme === 'light' ? '🌙' : '☀️'}
                </button>
            </div>
        </header>
    );
}
