import simpy
import random

class Intersection:
    def __init__(self, env):
        self.env = env
        self.light = simpy.Resource(env, capacity=1)  # Only one direction can be green
        self.green_duration = 30  # Time for green light
        self.yellow_duration = 5  # Time for yellow light
        self.cycle_time = self.green_duration + self.yellow_duration

    def run_traffic_light(self):
        """Simulate traffic light cycle."""
        while True:
            # Green for North-South
            print(f"Time {self.env.now}: North-South Green")
            yield self.env.timeout(self.green_duration)

            print(f"Time {self.env.now}: North-South Yellow")
            yield self.env.timeout(self.yellow_duration)

            # Green for East-West
            print(f"Time {self.env.now}: East-West Green")
            yield self.env.timeout(self.green_duration)

            print(f"Time {self.env.now}: East-West Yellow")
            yield self.env.timeout(self.yellow_duration)


class Vehicle:
    def __init__(self, env, name, intersection, direction):
        self.env = env
        self.name = name
        self.intersection = intersection
        self.direction = direction
        self.arrival_time = env.now
        env.process(self.run())

    def run(self):
        """Simulate a vehicle arriving and passing through the intersection."""
        print(f"Time {self.env.now}: {self.name} arrives at the intersection ({self.direction})")

        with self.intersection.light.request() as request:
            yield request  # Wait for green light

            print(f"Time {self.env.now}: {self.name} starts crossing ({self.direction})")
            yield self.env.timeout(2)  # Time to cross the intersection

            print(f"Time {self.env.now}: {self.name} crossed the intersection")


def vehicle_generator(env, intersection):
    """Generate vehicles arriving at the intersection."""
    vehicle_id = 0
    while True:
        yield env.timeout(random.expovariate(1 / 5))  # Random inter-arrival time (mean = 5)
        direction = random.choice(['North-South', 'East-West'])
        vehicle_id += 1
        Vehicle(env, f"Vehicle {vehicle_id}", intersection, direction)


# Simulation environment
env = simpy.Environment()

# Create intersection and start traffic light process
intersection = Intersection(env)
env.process(intersection.run_traffic_light())

# Start generating vehicles
env.process(vehicle_generator(env, intersection))

# Run the simulation for 120 seconds
env.run(until=120)
