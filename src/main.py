import carla
import os

host = os.getenv("CARLA_HOST", "localhost")
port = int(os.getenv("CARLA_PORT", 2000))

client = carla.Client(host, port)
client.set_timeout(10.0)

world = client.get_world()
print("✅ Connecté à CARLA :", world.get_map().name)
print("Hello world Im here")