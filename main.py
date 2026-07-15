"""
Healthcare Agent - Distributed Multi-Agent System

Main entry point for running the Streamlit UI locally.

For production deployment, use docker-compose:
   ./start-services.sh

For local development UI:
   python main.py
   # or
   streamlit run src/streamlit_app.py
"""

import sys
import os
import subprocess

def main():
    """Run the Streamlit UI application."""
    print("🏥 Starting Healthcare Agent Chat UI...")
    print("="*80)
    print("📍 UI will be available at: http://localhost:8501")
    print("📡 Connecting to API at:", os.getenv("SUPERVISOR_API_URL", "http://localhost:8000"))
    print("="*80)
    print("\n⚠️  Note: Make sure backend services are running:")
    print("   • Run: ./start-services.sh")
    print("   • Or ensure supervisor API is running on port 8000")
    print("\n🚀 Launching Streamlit...\n")
    
    # Run streamlit from src directory
    try:
        subprocess.run([
            "streamlit", "run", "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install:")
        print("   pip install streamlit")
        print("   # or")
        print("   uv pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
