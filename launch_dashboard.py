#!/usr/bin/env python3
"""
Spin36TB System Dashboard Launcher
Easy way to start the monitoring dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages for the dashboard"""
    requirements = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy'
    ]
    
    print("🔧 Installing dashboard requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "spin36TB_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return
    
    print("🚀 Launching Spin36TB System Dashboard...")
    print("📊 Dashboard will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(dashboard_path),
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    print("🎯 SPIN36TB SYSTEM DASHBOARD LAUNCHER")
    print("=" * 40)
    
    # Check if requirements are installed
    try:
        import streamlit
        import plotly
        print("✅ Dashboard requirements are installed")
    except ImportError:
        print("⚠️  Installing missing requirements...")
        install_requirements()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()