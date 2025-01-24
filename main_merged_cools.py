# Full simulation code (same as before with corrected `vehicle_cross`)
import simpy
import random
import numpy as np
import pandas as pd

random.seed(1024)
np.random.seed(1832)

# Constants
ROW_TIMES = {
    'Bakeri1': 2.675,
    'Bakeri2': 2.4666,
    'Besat1': 2.6818,
    'Besat2': 2.3243,
}

VEHICLE_PROBABILITIES = {
    'Bakeri1': [0.06, 0.82, 0.08, 0.04],
    'Bakeri2': [0.04, 0.75, 0.14, 0.07],
    'Besat1': [0.05, 0.80, 0.10, 0.05],
    'Besat2': [0.07, 0.77, 0.09, 0.07],
}

DIRECTION_PROBABILITIES = {
    'Bakeri1': {'left': 0.4, 'straight': 0.5, 'right': 0.1},
    'Bakeri2': {'left': 0.33, 'straight': 0.57, 'right': 0.1},
    'Besat1': {'left': 0.2, 'straight': 0.7, 'right': 0.1},
    'Besat2': {'left': 0.23, 'straight': 0.67, 'right': 0.1},
}

CROSSING_DISTRIBUTIONS = {
    'Bakeri1': lambda: np.random.poisson(8.21),
    'Bakeri2': lambda: np.random.normal(5.66, 2.08),
    'Besat1': lambda: min(10, 2.5 + np.random.lognormal(7.65, 6.42)),
    'Besat2': lambda: 2.5 + np.random.exponential(2.82),
}

ARRIVAL_DISTRIBUTIONS = {
    'Bakeri1': lambda: max(0.1, -0.5 + 22 * np.random.beta(0.971, 2.04)),
    'Bakeri2': lambda: max(0.1, -0.5 + 41 * np.random.beta(0.968, 3.44)),
    'Besat1': lambda: max(0.1, -0.5 + 23 * np.random.beta(0.634, 1.61)),
    'Besat2': lambda: max(0.1, -0.5 + 24 * np.random.beta(0.963, 1.99)),
}

TIME_MODIFIERS = {
    'Bus': {'row_time': 1.0, 'crossing_time': 1.0},
    'Car': {'row_time': 1.0, 'crossing_time': 1.0},
    'Van': {'row_time': 1.0, 'crossing_time': 1.0},
    'Pickup': {'row_time': 1.0, 'crossing_time': 1.0},
}

class IntersectionSimulation:
    def __init__(self, env, light_times, arrival_distributions):
        self.env = env
        self.light_times = light_times
        self.arrival_distributions = arrival_distributions
        self.queues = {street: {1: [], 2: [], 3: []} for street in arrival_distributions}
        self.light_state = {street: False for street in arrival_distributions}
        self.results = {street: [] for street in arrival_distributions}
        self.vehicle_count = 0

        self.env.process(self.traffic_light_cycle())
        for street in arrival_distributions:
            self.env.process(self.vehicle_arrival(street))

    def traffic_light_cycle(self):
        cycle_order = [(street, self.light_times['green'][street]) for street in self.light_times['green']]
        while True:
            for street, green_time in cycle_order:
                self.light_state[street] = True
                yield self.env.timeout(green_time)
                self.light_state[street] = False

    def vehicle_arrival(self, street):
        while True:
            inter_arrival_time = self.arrival_distributions[street]()
            yield self.env.timeout(inter_arrival_time)
            self.vehicle_count += 1
            vehicle_id = self.vehicle_count

            direction = random.choices(['left', 'straight', 'right'], weights=[
                DIRECTION_PROBABILITIES[street]['left'],
                DIRECTION_PROBABILITIES[street]['straight'],
                DIRECTION_PROBABILITIES[street]['right']
            ])[0]

            vehicle_type = random.choices(['Bus', 'Car', 'Van', 'Pickup'], weights=VEHICLE_PROBABILITIES[street])[0]

            if direction in ["left", "straight"]:
                queue_num = 1 if len(self.queues[street][1]) <= len(self.queues[street][2]) else 2
            else:
                queue_num = 3

            self.env.process(self.vehicle_cross(street, vehicle_id, direction, queue_num, vehicle_type))

    def vehicle_cross(self, street, vehicle_id, direction, queue_num, vehicle_type):
        """Simulates a vehicle crossing the intersection."""
        arrival_time = self.env.now
        queue_position = len(self.queues[street][queue_num])  # Vehicle's position in the queue
        row_index = queue_position + 1  # Row index is 1-based
        row_wait_time = (row_index - 1) * ROW_TIMES[street] * TIME_MODIFIERS[vehicle_type]['row_time']

        # Add the vehicle to the queue
        self.queues[street][queue_num].append(vehicle_id)
        yield self.env.timeout(row_wait_time)  # Wait time to move to the frontline

        # Wait for green light if not turning right
        red_light_wait_time = 0
        if direction != 'right':
            while not self.light_state[street]:
                yield self.env.timeout(1)
                red_light_wait_time += 1

        # Calculate crossing time (once)
        crossing_time = CROSSING_DISTRIBUTIONS[street]() * TIME_MODIFIERS[vehicle_type]['crossing_time']
        yield self.env.timeout(crossing_time)

        # Remove the vehicle from the queue
        self.queues[street][queue_num].remove(vehicle_id)
        total_time_spent = self.env.now - arrival_time



        # Record the total time spent for statistics
        self.results[street].append(total_time_spent)

        # Log details for this vehicle
        print(f"\n--- Vehicle {vehicle_id} Simulation ---")
        print(f"Street: {street}")
        print(f"Lane: {queue_num}, Row Index: {row_index}")
        print(f"Direction: {direction}")
        print(f"Vehicle Type: {vehicle_type}")
        print(f"Row Wait Time: {row_wait_time:.2f} seconds")
        print(f"Red Light Wait Time: {red_light_wait_time:.2f} seconds")
        print(f"Crossing Time: {crossing_time:.2f} seconds")
        print(f"\n--- Final Summary for Vehicle {vehicle_id} ---")
        print(f"Total Time Spent: {total_time_spent:.2f} seconds")
        print(f"  - Row Wait Time: {row_wait_time:.2f} seconds")
        print(f"  - Red Light Wait Time: {red_light_wait_time:.2f} seconds")
        print(f"  - Crossing Time: {crossing_time:.2f} seconds")
        print("------------------------------------------\n")

    def get_results(self):
        return {street: np.mean(times) if times else 0 for street, times in self.results.items()}

def run_simulation(scenarios):
    results = pd.DataFrame(columns=['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2'])
    for scenario in scenarios:
        env = simpy.Environment()
        sim = IntersectionSimulation(env, LIGHT_TIMES[scenario], ARRIVAL_DISTRIBUTIONS)
        env.run(until=1000)

        # Debugging: Inspect all collected results
        print("\n--- Debug: Results Content ---")
        for street, times in sim.results.items():
            print(f"{street}: {times}")

        avg_results = sim.get_results()
        avg_results['Average'] = np.mean(list(avg_results.values()))
        avg_results['Bakeri Average'] = np.mean([avg_results['Bakeri1'], avg_results['Bakeri2']])
        results.loc[scenario] = avg_results

        # Debug: Manual Averages
        print("\n--- Debug: Manual Averages ---")
        for street, times in sim.results.items():
            if times:
                manual_avg = np.mean(times)
                print(f"{street}: {manual_avg:.2f} (Manual Average)")
    print("\nTraffic Simulation Results")
    print(results)

    return results


LIGHT_TIMES = {
    3: {'green': {'Bakeri1': 40, 'Bakeri2': 30, 'Besat1': 25, 'Besat2': 25},
        'red': {'Bakeri1': 80, 'Bakeri2': 90, 'Besat1': 95, 'Besat2': 95}},
    4: {'green': {'Bakeri1': 40, 'Bakeri2': 35, 'Besat1': 30, 'Besat2': 25},
        'red': {'Bakeri1': 90, 'Bakeri2': 95, 'Besat1': 100, 'Besat2': 105}},
    7: {'green': {'Bakeri1': 30, 'Bakeri2': 25, 'Besat1': 30, 'Besat2': 30},
        'red': {'Bakeri1': 85, 'Bakeri2': 90, 'Besat1': 85, 'Besat2': 85}},
}


if __name__ == "__main__":
    scenarios = [3]
    run_simulation(scenarios)