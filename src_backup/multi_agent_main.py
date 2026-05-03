"""
Multi-Agent Autonomous Driving System - Main Entry Point

This module coordinates multiple AI agents for autonomous driving:
- Perception Agent: Object detection and scene understanding
- Prediction Agent: Behavior prediction for other road users
- Decision-Making Agent: High-level driving decisions
- Planning Agent: Trajectory planning and path generation
- Control Agent: Low-level vehicle control

The system integrates with CARLA simulator for realistic testing.
"""

import argparse
import time
import threading
from typing import Dict, List, Any

# Import multi-agent system
from agents.base_agent import MessageBus, EnvironmentState, VehicleState
from agents.perception_agent import PerceptionAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionMakingAgent
from agents.planning_agent import PlanningAgent
from agents.control_agent import ControlAgent

# Import CARLA simulation (fallback to simple simulation if not available)
try:
    from envs.carla_simulation import CarlaSimulation
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("⚠️ CARLA not available, using simplified simulation")


class MultiAgentDrivingSystem:
    """Main multi-agent autonomous driving system."""

    def __init__(self, carla_host: str = "localhost", carla_port: int = 2000,
                 simulation_duration: int = 300, use_carla: bool = True):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.simulation_duration = simulation_duration
        self.use_carla = use_carla

        # Initialize message bus
        self.message_bus = MessageBus()

        # Initialize agents
        self.agents = {}
        self._initialize_agents()

        # Simulation state
        self.simulation = None
        self.running = False
        self.start_time = 0.0
        self._shutdown_complete = False

    def _initialize_agents(self) -> None:
        """Initialize all agents and register them."""
        print("🤖 Initializing Multi-Agent System...")

        # Create agents
        self.agents['perception'] = PerceptionAgent(self.message_bus)
        self.agents['prediction'] = PredictionAgent(self.message_bus)
        self.agents['decision'] = DecisionMakingAgent(self.message_bus)
        self.agents['planning'] = PlanningAgent(self.message_bus)
        self.agents['control'] = ControlAgent(self.message_bus)

        # Initialize agents
        for agent_name, agent in self.agents.items():
            if not agent.initialize():
                print(f"❌ Failed to initialize {agent_name} agent")
                return

        print("✅ All agents initialized successfully")

    def _initialize_simulation(self) -> bool:
        """Initialize CARLA simulation or fallback."""
        if not self.use_carla:
            print("⚠️ Running in no-CARLA mode with simplified simulation")
            self.simulation = None
            return True

        if CARLA_AVAILABLE:
            try:
                self.simulation = CarlaSimulation(
                    host=self.carla_host,
                    port=self.carla_port,
                    map_name="Town03"
                )
                print("✅ CARLA simulation initialized")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize CARLA: {e}")
                return False
        else:
            print("⚠️ CARLA is not available, using simplified simulation")
            self.simulation = None
            return True

    def _create_environment_state(self) -> EnvironmentState:
        """Create current environment state from simulation."""
        if self.simulation:
            # Get state from CARLA simulation
            sim_state = self.simulation.get_state()

            # Convert to our environment state format
            ego_vehicle = VehicleState(
                position=(0, 0, 0),  # Would need to get from CARLA
                velocity=(0, 0, 0),
                acceleration=(0, 0, 0),
                orientation=(0, 0, 0),
                angular_velocity=(0, 0, 0)
            )

            env_state = EnvironmentState(
                timestamp=time.time() - self.start_time,
                ego_vehicle=ego_vehicle,
                other_vehicles=[],  # Would populate from CARLA
                pedestrians=[],     # Would populate from CARLA
                traffic_lights=[],
                lane_info={},
                weather_conditions={}
            )

            # Add camera data if available
            if 'camera' in sim_state:
                env_state.ego_vehicle.camera = sim_state['camera']

            return env_state
        else:
            # Create dummy state for testing
            ego_vehicle = VehicleState(
                position=(0, 0, 0),
                velocity=(10, 0, 0),  # 10 m/s forward
                acceleration=(0, 0, 0),
                orientation=(0, 0, 0),
                angular_velocity=(0, 0, 0)
            )

            return EnvironmentState(
                timestamp=time.time() - self.start_time,
                ego_vehicle=ego_vehicle,
                other_vehicles=[],
                pedestrians=[{'position': (0, 20, 0), 'velocity': (0, 0, 0)}],  # Dummy pedestrian
                traffic_lights=[],
                lane_info={},
                weather_conditions={'condition': 'clear'}
            )

    def _run_agent_cycle(self, env_state: EnvironmentState) -> None:
        """Run one cycle of all agents."""
        # Process agents in order
        agent_order = ['perception', 'prediction', 'decision', 'planning', 'control']

        for agent_name in agent_order:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                try:
                    agent.process(env_state)
                except Exception as e:
                    print(f"❌ Error in {agent_name} agent: {e}")

    def _log_system_status(self) -> None:
        """Log current system status."""
        status_info = {}
        for agent_name, agent in self.agents.items():
            status_info[agent_name] = agent.get_status()

        print(f"📊 System Status [t={time.time() - self.start_time:.1f}s]:")
        for agent_name, status in status_info.items():
            state = status.get('state', 'unknown')
            print(f"  • {agent_name}: {state}")

    def run(self) -> None:
        """Run the multi-agent autonomous driving system."""
        print("🚗 Starting Multi-Agent Autonomous Driving System...")

        # Initialize simulation
        if not self._initialize_simulation():
            print("❌ Failed to initialize simulation")
            return

        self.running = True
        self.start_time = time.time()

        try:
            # Reset simulation
            if self.simulation:
                state = self.simulation.reset()
                print("✅ Simulation reset complete")

            # Main simulation loop
            frame_count = 0
            last_log_time = time.time()

            while self.running and frame_count < self.simulation_duration:
                current_time = time.time()

                # Create environment state
                env_state = self._create_environment_state()

                # Run agent cycle
                self._run_agent_cycle(env_state)

                # Get control command from control agent
                control_agent = self.agents.get('control')
                if control_agent:
                    control_command = control_agent.get_current_command()

                    # Apply control to simulation
                    if self.simulation and hasattr(control_command, 'throttle'):
                        self.simulation.apply_ego_control(
                            throttle=control_command.throttle,
                            steer=control_command.steer
                        )

                # Tick simulation
                if self.simulation:
                    self.simulation.tick()

                # Periodic logging
                if current_time - last_log_time > 2.0:  # Log every 2 seconds
                    self._log_system_status()
                    last_log_time = current_time

                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                frame_count += 1

            print("🎯 Simulation completed successfully")

        except KeyboardInterrupt:
            print("⏹️ Simulation interrupted by user")
        except Exception as e:
            print(f"❌ Simulation error: {e}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        if self._shutdown_complete:
            return

        self._shutdown_complete = True
        print("🔄 Shutting down Multi-Agent System...")

        self.running = False

        # Shutdown agents
        for agent in self.agents.values():
            agent.shutdown()

        # Close simulation
        if self.simulation:
            self.simulation.close()

        print("✅ System shutdown complete")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Autonomous Driving System")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in frames")
    parser.add_argument("--no-carla", action="store_true", help="Run without CARLA (simulation only)")

    args = parser.parse_args()

    # Create and run system
    system = MultiAgentDrivingSystem(
        carla_host=args.host,
        carla_port=args.port,
        simulation_duration=args.duration,
        use_carla=not args.no_carla
    )

    try:
        system.run()
    except KeyboardInterrupt:
        print("⏹️ System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()