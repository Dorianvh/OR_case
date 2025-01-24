import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    'Bakeri1': lambda: max(0.1, np.random.poisson(8.21)),
    'Bakeri2': lambda: max(0.1, np.random.normal(5.66, 2.08)),
    'Besat1': lambda: max(0.1, min(10, 2.5 + np.random.lognormal(7.65, 6.42))),
    'Besat2': lambda: max(0.1, 2.5 + np.random.exponential(2.82)),
}

ARRIVAL_DISTRIBUTIONS = {
    'Bakeri1': lambda: max(0.1, -0.5 + 22 * np.random.beta(0.971, 2.04)),
    'Bakeri2': lambda: max(0.1, -0.5 + 41 * np.random.beta(0.968, 3.44)),
    'Besat1': lambda: max(0.1, -0.5 + 23 * np.random.beta(0.634, 1.61)),
    'Besat2': lambda: max(0.1, -0.5 + 24 * np.random.beta(0.963, 1.99)),
}

TIME_MODIFIERS = {
    'Bus': {'row_time': 1.5, 'crossing_time': 1.8},
    'Car': {'row_time': 1.0, 'crossing_time': 1.0},
    'Van': {'row_time': 1.2, 'crossing_time': 1.2},
    'Pickup': {'row_time': 1.1, 'crossing_time': 1.1},
}

THRESHOLDS = {
    'Bakeri1': 40,
    'Bakeri2': 30,
    'Besat1': 25,
    'Besat2': 25,
}


# SPSA Parameters
T = 10000  # Total iterations
epsilon = 0.1  # Learning rate
delta_0 = 5  # Initial perturbation magnitude
delta_decay_rate = 0.101  # Decay rate for delta
PHI_MIN = 20  # Minimum green light duration

class IntersectionSimulation:
    def __init__(self, env, thresholds, arrival_distributions):
        self.env = env
        self.thresholds = thresholds
        self.arrival_distributions = arrival_distributions
        self.queues = {street: {1: [], 2: [], 3: []} for street in arrival_distributions}
        self.light_state = {street: {'is_green': False, 'time_since_green': 0, 'holding_time': 0} for street in arrival_distributions}
        self.results = {street: [] for street in arrival_distributions}
        self.vehicle_count = 0

        for street in arrival_distributions:
            self.env.process(self.vehicle_arrival(street))
            self.env.process(self.evaluate_lights(street))

    def evaluate_lights(self, street):
        """Evaluate the light state based on the integral of time over cars."""
        while True:
            current_state = self.light_state[street]
            current_state['time_since_green'] += 1

            if not current_state['is_green']:
                # Update holding time (integral of cars over time)
                holding_time = sum(len(queue) for queue in self.queues[street].values())
                current_state['holding_time'] += holding_time

                # Check if threshold is met and minimum phase time is satisfied
                if current_state['holding_time'] >= self.thresholds[street] and current_state['time_since_green'] >= PHI_MIN:
                    current_state['is_green'] = True
                    current_state['time_since_green'] = 0
                    current_state['holding_time'] = 0
                    yield self.env.timeout(10)  # Green light duration
                    current_state['is_green'] = False

            yield self.env.timeout(1)

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
        arrival_time = self.env.now
        queue_position = len(self.queues[street][queue_num])
        row_index = queue_position + 1
        row_wait_time = (row_index - 1) * ROW_TIMES[street] * TIME_MODIFIERS[vehicle_type]['row_time']

        self.queues[street][queue_num].append(vehicle_id)
        yield self.env.timeout(row_wait_time)

        red_light_wait_time = 0
        if direction != 'right':
            while not self.light_state[street]['is_green']:
                yield self.env.timeout(1)
                red_light_wait_time += 1

        crossing_time = CROSSING_DISTRIBUTIONS[street]() * TIME_MODIFIERS[vehicle_type]['crossing_time']
        yield self.env.timeout(crossing_time)

        self.queues[street][queue_num].remove(vehicle_id)
        total_time_spent = self.env.now - arrival_time

        self.results[street].append(total_time_spent)

    def get_results(self):
        return {street: np.mean(times) if times else 0 for street, times in self.results.items()}

def run_simulation(thresholds):
    env = simpy.Environment()
    sim = IntersectionSimulation(env, thresholds, ARRIVAL_DISTRIBUTIONS)
    env.run(until=1000)
    return sim.get_results()

def simulate_cost(theta):
    thresholds = {street: max(5, theta[i]) for i, street in enumerate(THRESHOLDS.keys())}
    results = run_simulation(thresholds)
    street_costs = [results[street] for street in THRESHOLDS.keys()]
    return np.mean(street_costs), street_costs

def spsa_optimization(theta, T, epsilon, delta_0, delta_decay_rate):
    costs = []
    street_cost_history = {street: [] for street in THRESHOLDS.keys()}
    theta_history = []

    for t in range(1, T + 1):
        delta = delta_0 / (t ** delta_decay_rate)  # Decreasing delta over iterations
        delta_vector = np.random.choice([-1, 1], size=len(theta))

        theta_plus = theta + delta * delta_vector
        theta_minus = theta - delta * delta_vector

        J_plus, street_cost_plus = simulate_cost(theta_plus)
        J_minus, street_cost_minus = simulate_cost(theta_minus)

        g_hat = (J_plus - J_minus) / (2 * delta * delta_vector)

        theta = theta - epsilon * g_hat
        theta = np.maximum(theta, 5)  # Enforce minimum threshold values

        current_cost, street_costs = simulate_cost(theta)
        costs.append(current_cost)
        for i, street in enumerate(THRESHOLDS.keys()):
            street_cost_history[street].append(street_costs[i])
        theta_history.append(theta.copy())

        if t % 10 == 0 or t == T:
            print(f"Iteration {t}: Cost = {current_cost:.2f}, Theta = {theta}")

    return theta, costs, theta_history, street_cost_history

if __name__ == "__main__":
    initial_theta = np.array(list(THRESHOLDS.values()))
    final_theta, cost_history, theta_history, street_cost_history = spsa_optimization(initial_theta, T, epsilon, delta_0, delta_decay_rate)

    print("\nOptimization Complete")
    print("Final Thresholds:", final_theta)

    plt.figure(figsize=(10, 5))
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Average Waiting Time")
    plt.title("SPSA Optimization - Cost History")
    plt.show()

    plt.figure(figsize=(10, 5))
    theta_history = np.array(theta_history)
    for i, street in enumerate(THRESHOLDS.keys()):
        plt.plot(theta_history[:, i], label=street)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold Values")
    plt.title("Theta Updates Over Iterations")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    for street, costs in street_cost_history.items():
        plt.plot(costs, label=f"{street} Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Street-Specific Waiting Times")
    plt.title("Street-Specific Cost History")
    plt.legend()
    plt.show()
