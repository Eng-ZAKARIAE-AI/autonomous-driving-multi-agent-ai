import carla
import yaml
import time
from pathlib import Path

def test_connection():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ config/config.yaml not found")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    host = config['carla']['host']
    port = config['carla']['port']
    timeout = config['carla']['timeout']

    print(f"🔄 Connecting to CARLA at {host}:{port} (timeout={timeout}s)...")
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        version = client.get_server_version()
        print(f"✅ Connected! CARLA Server Version: {version}")
        
        world = client.get_world()
        print(f"🌍 Current Map: {world.get_map().name}")
        
        # Test synchronous mode if configured
        if config['carla'].get('synchronous', False):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = config['carla'].get('fixed_delta_seconds', 0.05)
            world.apply_settings(settings)
            print("✅ Synchronous mode enabled")
            
            world.tick()
            print("✅ World ticked successfully")
            
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("🔄 Restored asynchronous mode")
            
        print("\n🚀 CARLA Connection Test Passed!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Tips:")
        print("1. Make sure CARLA server is running.")
        print("2. Check if host and port are correct in config/config.yaml.")
        print("3. Increase timeout if the server is slow.")

if __name__ == "__main__":
    test_connection()
