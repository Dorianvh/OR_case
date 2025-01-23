#!/usr/bin/env python

import simpy
import random
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# --------------------------------------------------
# 1) SIMULATION CONSTANTS AND PROBABILITIES
# --------------------------------------------------

random.seed(1024)
np.random.seed(1832)

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
    'Bus':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Car':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Van':    {'row_time': 1.0, 'crossing_time': 1.0},
    'Pickup': {'row_time': 1.0, 'crossing_time': 1.0},
}

LIGHT_TIMES = {
    3: {
        'green': {'Bakeri1': 40, 'Bakeri2': 30, 'Besat1': 25, 'Besat2': 25},
        'red':   {'Bakeri1': 80, 'Bakeri2': 90, 'Besat1': 95, 'Besat2': 95}
    },
    4: {
        'green': {'Bakeri1': 40, 'Bakeri2': 35, 'Besat1': 30, 'Besat2': 25},
        'red':   {'Bakeri1': 90, 'Bakeri2': 95, 'Besat1': 100, 'Besat2': 105}
    },
    7: {
        'green': {'Bakeri1': 30, 'Bakeri2': 25, 'Besat1': 30, 'Besat2': 30},
        'red':   {'Bakeri1': 85, 'Bakeri2': 90, 'Besat1': 85, 'Besat2': 85}
    },
}


# --------------------------------------------------
# 2) SIMULATION CLASS
# --------------------------------------------------

class IntersectionSimulation:
    """
    Simulates an intersection with four streets. Each street has 3 lanes and one traffic light.
    Vehicles arrive with specific distributions, join a queue, wait for green, then cross.
    """

    def __init__(self, env, light_times, arrival_distributions):
        """
        :param env: SimPy Environment
        :param light_times: dict specifying green and red durations for each street
        :param arrival_distributions: dict specifying vehicle inter-arrival distributions per street
        """
        self.env = env
        self.light_times = light_times
        self.arrival_distributions = arrival_distributions

        # Queues: For each street, we have 3 lanes: {1, 2, 3}
        self.queues = {
            street: {1: [], 2: [], 3: []}
            for street in arrival_distributions
        }

        # Light state for each street: True = Green, False = Red
        self.light_state = {street: False for street in arrival_distributions}

        # Track times for each street
        self.results = {street: [] for street in arrival_distributions}

        self.vehicle_count = 0

        # Start traffic light cycle
        self.env.process(self.traffic_light_cycle())

        # Start vehicle arrival processes
        for street in arrival_distributions:
            self.env.process(self.vehicle_arrival(street))

    def traffic_light_cycle(self):
        """
        Rotates the green light among the streets. Each has a designated 'green' time.
        After that, we switch it to red and move to the next street in cycle_order.
        """
        cycle_order = [(s, self.light_times['green'][s]) for s in self.light_times['green']]

        while True:
            for street, green_time in cycle_order:
                self.light_state[street] = True
                yield self.env.timeout(green_time)
                self.light_state[street] = False

    def vehicle_arrival(self, street):
        """
        Generates vehicles for a given street according to the arrival distribution.
        """
        while True:
            # Wait for the next inter-arrival
            inter_arrival_time = self.arrival_distributions[street]()
            yield self.env.timeout(inter_arrival_time)

            # Assign vehicle ID
            self.vehicle_count += 1
            vehicle_id = self.vehicle_count

            # Determine direction (left, straight, right)
            direction = random.choices(
                ['left', 'straight', 'right'],
                weights=[
                    DIRECTION_PROBABILITIES[street]['left'],
                    DIRECTION_PROBABILITIES[street]['straight'],
                    DIRECTION_PROBABILITIES[street]['right']
                ],
                k=1
            )[0]

            # Determine vehicle type
            vehicle_type = random.choices(
                ['Bus', 'Car', 'Van', 'Pickup'],
                weights=VEHICLE_PROBABILITIES[street],
                k=1
            )[0]

            # Simple logic to decide queue/lane:
            # Use lanes 1 or 2 if direction is left or straight, else lane 3 for right turns
            if direction in ["left", "straight"]:
                # Put the vehicle in the lane with the shorter queue (lane1 or lane2)
                lane1_len = len(self.queues[street][1])
                lane2_len = len(self.queues[street][2])
                queue_num = 1 if lane1_len <= lane2_len else 2
            else:
                queue_num = 3

            # Start vehicle crossing process
            self.env.process(
                self.vehicle_cross(street, vehicle_id, direction, queue_num, vehicle_type)
            )

    def vehicle_cross(self, street, vehicle_id, direction, queue_num, vehicle_type):
        """
        Simulates a single vehicle's travel through the intersection:
          1) Wait in queue (row wait time).
          2) Wait for green light (unless turning right).
          3) Cross the intersection.
        """
        arrival_time = self.env.now

        # Position in the queue
        queue_position = len(self.queues[street][queue_num])
        row_index = queue_position + 1  # 1-based index

        # Time waiting to move from row_index N to the front (index 1)
        row_wait_time = ((row_index - 1)
                         * ROW_TIMES[street]
                         * TIME_MODIFIERS[vehicle_type]['row_time'])

        # Add vehicle to queue
        self.queues[street][queue_num].append(vehicle_id)

        # Wait that row time
        yield self.env.timeout(row_wait_time)

        # If direction != 'right', must wait for green
        red_light_wait_time = 0
        if direction != 'right':
            while not self.light_state[street]:
                yield self.env.timeout(1)
                red_light_wait_time += 1

        # Crossing time
        crossing_time = (CROSSING_DISTRIBUTIONS[street]()
                         * TIME_MODIFIERS[vehicle_type]['crossing_time'])
        yield self.env.timeout(crossing_time)

        # Remove vehicle from queue
        self.queues[street][queue_num].remove(vehicle_id)

        total_time_spent = self.env.now - arrival_time
        self.results[street].append(total_time_spent)

        # Debug print (optional, remove or comment out to reduce console spam)
        print(f"\n--- Vehicle {vehicle_id} Simulation ---")
        print(f"Street: {street}")
        print(f"Lane: {queue_num} | Row Index: {row_index}")
        print(f"Direction: {direction}")
        print(f"Vehicle Type: {vehicle_type}")
        print(f"Row Wait Time: {row_wait_time:.2f} seconds")
        print(f"Red Light Wait Time: {red_light_wait_time:.2f} seconds")
        print(f"Crossing Time: {crossing_time:.2f} seconds")
        print(f"Total Time Spent: {total_time_spent:.2f} seconds")
        print("------------------------------------------\n")

    def get_results(self):
        """
        :return: Dict of {street: avg_time_spent} across all vehicles for that street
        """
        return {
            street: (np.mean(times) if times else 0)
            for street, times in self.results.items()
        }


# --------------------------------------------------
# 3) TKINTER-BASED GUI CLASS
# --------------------------------------------------

class IntersectionGUI:
    """
    A simple Tkinter GUI that runs the IntersectionSimulation step-by-step.
    Displays each street's traffic light status and queue lengths.
    """

    def __init__(self, master, light_times, arrival_distributions, run_time=200):
        """
        :param master: The Tk root window.
        :param light_times: Dictionary of light times (e.g., LIGHT_TIMES[3])
        :param arrival_distributions: Dictionary of arrival distributions
        :param run_time: How long (simpy time) to run the simulation
        """
        self.master = master
        self.master.title("Intersection Simulation")

        # Create SimPy environment and the simulation itself
        self.env = simpy.Environment()
        self.sim = IntersectionSimulation(self.env, light_times, arrival_distributions)

        # We'll stop stepping once self.env.now >= run_time
        self.run_time = run_time

        # Create Tkinter labels to show state
        self.street_labels = {}  # for traffic lights
        self.lane_labels = {}    # for lane queue lengths
        self.setup_gui()

        # Start stepping the simulation
        self.step_simulation()

    def setup_gui(self):
        """
        Builds all the label widgets to display the queue lengths & light states.
        """
        row_idx = 0
        for street in self.sim.arrival_distributions.keys():
            frame = ttk.LabelFrame(self.master, text=street, padding=10)
            frame.grid(row=row_idx, column=0, padx=10, pady=10, sticky="ew")

            # Traffic light label
            light_label = ttk.Label(frame, text="Light: RED", foreground="red",
                                    font=("TkDefaultFont", 10, "bold"))
            light_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.street_labels[street] = light_label

            # Lane labels
            for lane in range(1, 4):
                lbl = ttk.Label(frame, text=f"Lane {lane} queue: 0")
                lbl.grid(row=lane, column=0, padx=5, pady=2, sticky="w")
                self.lane_labels[(street, lane)] = lbl

            row_idx += 1

    def step_simulation(self):
        """
        Advance the simulation by one event and update the GUI.
        Schedule the next step if run_time not exceeded and events remain.
        """
        if self.env.now >= self.run_time:
            # Simulation time is up
            return

        try:
            # Advance the simulation by the next event
            self.env.step()
        except simpy.core.EmptySchedule:
            # No more events in the queue
            return

        # Update Tk labels with the current simulation state
        self.update_gui()

        # Schedule another step in 100 ms (real time)
        self.master.after(10, self.step_simulation)

    def update_gui(self):
        """
        Refresh traffic light labels and queue length labels.
        """
        # Update traffic lights
        for street, is_green in self.sim.light_state.items():
            if is_green:
                self.street_labels[street].config(text="Light: GREEN", foreground="green")
            else:
                self.street_labels[street].config(text="Light: RED", foreground="red")

        # Update queue lengths
        for street, lanes_dict in self.sim.queues.items():
            for lane_num, queue_list in lanes_dict.items():
                q_len = len(queue_list)
                self.lane_labels[(street, lane_num)].config(
                    text=f"Lane {lane_num} queue: {q_len}"
                )


# --------------------------------------------------
# 4) MAIN FUNCTION TO LAUNCH THE GUI
# --------------------------------------------------

def main():
    """
    Runs the simulation scenario with a Tkinter GUI. After the GUI closes,
    prints the average time spent per street.
    """
    # Choose one scenario (3, 4, or 7) from LIGHT_TIMES
    scenario = 3

    root = tk.Tk()
    app = IntersectionGUI(root, LIGHT_TIMES[scenario], ARRIVAL_DISTRIBUTIONS, run_time=200)
    root.mainloop()

    # Once the GUI is closed, we can retrieve & print simulation results
    results = app.sim.get_results()
    print("\nSimulation complete.")
    print("Average time spent (by street):")
    for street, avg_time in results.items():
        print(f"  {street}: {avg_time:.2f} seconds")


if __name__ == "__main__":
    main()
