import matplotlib.pyplot as plt
import simpy
import random
import numpy as np
import tkinter as tk
from tkinter import ttk

# --------------------------------------------------
# 1) SIMULATION CONSTANTS & SETUP
# --------------------------------------------------

random.seed(1024)
np.random.seed(1832)

# -----------------------------
# SOTL PARAMETERS
# -----------------------------

THRESHOLD = 5  # θ
PHI_MIN = 5     # minimum green phase
OMEGA = 5       # distance to detect a crossing platoon (simplified: see below)
MU = 8          # break large platoons if > MU cars waiting on red


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
    'Bus':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Car':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Van':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Pickup': {'row_time': 1.0, 'crossing_time': 1.0},
}

LIGHT_TIMES = {
    3: {
        'green': {'Bakeri1': 40, 'Bakeri2': 30, 'Besat1': 25, 'Besat2': 25},
        'red':   {'Bakeri1': 80, 'Bakeri2': 90, 'Besat1': 95, 'Besat2': 95}
    }
}


# --------------------------------------------------
# 2) SIMULATION LOGIC
# --------------------------------------------------

class IntersectionSimulation:
    def __init__(self, env, light_times, arrival_distributions, thresholds):
        self.env = env
        self.light_times = light_times
        self.arrival_distributions = arrival_distributions
        self.thresholds = thresholds

        self.queues = {
            street: {1: [], 2: [], 3: []}
            for street in arrival_distributions
        }
        self.light_state = {street: False for street in arrival_distributions}
        self.results = {street: [] for street in arrival_distributions}
        self.vehicle_count = 0
        self.average_time = 0
        self.arrival_history = {street: [] for street in arrival_distributions}

        self.env.process(self.traffic_light_cycle())

        for street in arrival_distributions:
            self.env.process(self.vehicle_arrival(street))

    def traffic_light_cycle(self):
        """Set Bakeri1 to green initially and switch to other streets based on queue length."""
        self.light_state['Bakeri1'] = True
        while True:
            for street, lanes in self.queues.items():
                # Check if total queue length (sum of all lanes in a street) exceeds threshold
                total_queue_length = sum(len(queue) for queue in lanes.values())

                if total_queue_length >= self.thresholds[street]:
                    # Turn all lights red
                    self.light_state = {s: False for s in self.light_state}
                    # Set the current street light to green
                    self.light_state[street] = True
                    # Ensure the green light stays on for at least PHI_MIN seconds
                    green_light_time = 0
                    while sum(len(queue) for queue in self.queues[street].values()) >= self.thresholds[street] or green_light_time < PHI_MIN:
                        yield self.env.timeout(1)
                        green_light_time += 1

            yield self.env.timeout(1)

    def vehicle_arrival(self, street):
        while True:
            inter_arrival_time = self.arrival_distributions[street]()
            self.arrival_history[street].append((self.env.now, inter_arrival_time))
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
        queue_position = len(self.queues[street][queue_num])  # Vehicle's position in queue
        row_index = queue_position + 1
        row_wait_time = (row_index - 1) * ROW_TIMES[street] * TIME_MODIFIERS[vehicle_type]['row_time']

        # Add vehicle to queue
        self.queues[street][queue_num].append(vehicle_id)
        yield self.env.timeout(row_wait_time)  # Wait time to move to the first row

        # Wait for green light if not turning right
        red_light_wait_time = 0
        if direction != 'right':
            while not self.light_state[street]:

                yield self.env.timeout(1)
                red_light_wait_time += 1
        else:
            if not self.light_state[street]:  # If red light but turning right
                pass
        # Calculate crossing time
        crossing_time = CROSSING_DISTRIBUTIONS[street]() * TIME_MODIFIERS[vehicle_type]['crossing_time']
        yield self.env.timeout(crossing_time)

        # Remove vehicle from queue
        self.queues[street][queue_num].remove(vehicle_id)
        total_time_spent = self.env.now - arrival_time

        # Record total time
        self.results[street].append(total_time_spent)


    def get_average_time(self):
        total_times = [time for times in self.results.values() for time in times]
        self.average_time = np.mean(total_times) if total_times else 0
        return self.average_time

    def get_results(self):
        return {
            street: (np.mean(times) if times else 0)
            for street, times in self.results.items()
        }

    def update_parameters(self, arrival_distributions=None, vehicle_probabilities=None, crossing_distributions=None):
        """Update travel times, vehicle probabilities, and crossing probabilities dynamically."""
        if arrival_distributions:
            for street, func in arrival_distributions.items():
                self.arrival_distributions[street] = func

        if vehicle_probabilities:
            for street, probabilities in vehicle_probabilities.items():
                VEHICLE_PROBABILITIES[street] = probabilities

        if crossing_distributions:
            for street, func in crossing_distributions.items():
                CROSSING_DISTRIBUTIONS[street] = func

    def dynamic_update(self, iteration):
        """Dynamically update travel time characteristics during the simulation."""
        if iteration % 100 == 0:  # Update every 100 iterations
            factor = 1.2  # Incrementally increase by 20% at iteration 100
            
            # Update arrival distributions
            new_arrival_distributions = {
                'Bakeri1': lambda: max(0.1, -0.5 + 22 * np.random.beta(0.971 * factor, 2.04 / factor)),
                'Bakeri2': lambda: max(0.1, -0.5 + 41 * np.random.beta(0.968 * factor, 3.44 / factor)),
                'Besat1': lambda: max(0.1, -0.5 + 23 * np.random.beta(0.634 * factor, 1.61 / factor)),
                'Besat2': lambda: max(0.1, -0.5 + 24 * np.random.beta(0.963 * factor, 1.99 / factor))
            }

            # Update crossing distributions
            new_crossing_distributions = {
                'Bakeri1': lambda: max(0.1, np.random.poisson(8.21 * factor)),
                'Bakeri2': lambda: max(0.1, np.random.normal(5.66 * factor, 2.08)),
                'Besat1': lambda: max(0.1, min(10, 2.5 + np.random.lognormal(7.65 * factor, 6.42))),
                'Besat2': lambda: max(0.1, 2.5 + np.random.exponential(2.82 * factor))
            }

            # Update vehicle probabilities (example: increase Bus probability slightly)
            new_vehicle_probabilities = {
                'Bakeri1': [0.07, 0.80, 0.08, 0.05],
                'Bakeri2': [0.05, 0.70, 0.15, 0.10],
                'Besat1': [0.06, 0.75, 0.11, 0.08],
                'Besat2': [0.08, 0.73, 0.12, 0.07]
            }

            self.update_parameters(
                arrival_distributions=new_arrival_distributions,
                vehicle_probabilities=new_vehicle_probabilities,
                crossing_distributions=new_crossing_distributions
            )
            print(f"Updated distributions dynamically at iteration {iteration} with factor: {factor:.2f}")



    def plot_arrival_distributions(self):
        """Plot the arrival distributions to show changes over time."""
        plt.figure(figsize=(12, 6))
        for street, history in self.arrival_history.items():
            times, intervals = zip(*history) if history else ([], [])
            plt.plot(times, intervals, label=street)

        plt.xlabel("Simulation Time")
        plt.ylabel("Arrival Interval")
        plt.title("Arrival Distribution Changes Over Time")
        plt.legend()
        plt.show()

   

# --------------------------------------------------
# 5) SPSA OPTIMIZATION
# --------------------------------------------------

class SPSAOptimizer:
    def __init__(self, sim_instance, initial_theta, a=0.05, c=0.05, alpha=0, gamma=0, max_iter=500):
        self.sim = sim_instance  # Pass the simulation instance here
        self.theta = initial_theta
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.theta_history = []
        self.loss_history = []

    def optimize(self):
        ak = self.a / (np.arange(1, self.max_iter + 1) ** self.alpha)
        ck = self.c / (np.arange(1, self.max_iter + 1) ** self.gamma)

        for k in range(self.max_iter):
            delta = 2 * np.random.randint(2, size=len(self.theta)) - 1
            theta_plus = self.theta + ck[k] * delta
            theta_minus = self.theta - ck[k] * delta

            # Evaluate the loss for theta_plus and theta_minus
            loss_plus = self.evaluate(theta_plus)
            loss_minus = self.evaluate(theta_minus)

            # Compute the gradient estimate
            gk = (loss_plus - loss_minus) / (2 * ck[k] * delta)

            # Update theta with a lower bound of 0
            self.theta = np.maximum(2, self.theta - ak[k] * gk)

            # Store the current theta and loss
            self.theta_history.append(self.theta.copy())
            self.loss_history.append(self.evaluate(self.theta))

            # Trigger dynamic update every 30 iterations
            if (k + 1) % 100 == 0:  # Ensure updates every 30 iterations
                self.sim.dynamic_update(k + 1)

            # Print the current iteration, theta, and loss
            print(f"Iteration {k+1}/{self.max_iter}, theta: {self.theta}, loss: {self.loss_history[-1]}")

        return self.theta

    def evaluate(self, theta):
        thresholds = {
            'Bakeri1': theta[0],
            'Bakeri2': theta[1],
            'Besat1': theta[2],
            'Besat2': theta[3]
        }
        env = simpy.Environment()
        self.sim = IntersectionSimulation(env, LIGHT_TIMES[3], ARRIVAL_DISTRIBUTIONS, thresholds)  # Reinitialize sim
        env.run(until=3600)
        average_time = self.sim.get_average_time()

        # Print the evaluated theta and the resulting average time
        print(f"Evaluating theta: {theta}, average time: {average_time}")

        return average_time


    def plot_results(self):
        # Plot theta over iterations
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.theta_history)
        plt.xlabel('Iteration')
        plt.ylabel('Theta')
        plt.title('Theta over Iterations')

        # Plot loss over iterations
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')

        plt.tight_layout()
        plt.show()

# --------------------------------------------------
# 6) MAIN
# --------------------------------------------------

def main():
    env = simpy.Environment()
    sim_instance = IntersectionSimulation(env, LIGHT_TIMES[3], ARRIVAL_DISTRIBUTIONS, [15, 15, 15, 15])
    initial_theta = [15, 15, 15, 15]  # Initial guess for thresholds for each street
    optimizer = SPSAOptimizer(sim_instance, initial_theta)
    optimal_theta = optimizer.optimize()
    print(f"Optimal THRESHOLDS: {optimal_theta}")
    optimizer.plot_results()




if __name__ == "__main__":
    main()
