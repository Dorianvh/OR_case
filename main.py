import random
import numpy as np
import pandas as pd

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

LIGHT_TIMES = {
    3: {'green': {'Bakeri1': 40, 'Bakeri2': 30, 'Besat1': 25, 'Besat2': 25},
        'red': {'Bakeri1': 80, 'Bakeri2': 90, 'Besat1': 95, 'Besat2': 95}},
    4: {'green': {'Bakeri1': 40, 'Bakeri2': 35, 'Besat1': 30, 'Besat2': 25},
        'red': {'Bakeri1': 90, 'Bakeri2': 95, 'Besat1': 100, 'Besat2': 105}},
    7: {'green': {'Bakeri1': 30, 'Bakeri2': 25, 'Besat1': 30, 'Besat2': 30},
        'red': {'Bakeri1': 85, 'Bakeri2': 90, 'Besat1': 85, 'Besat2': 85}},
}

CROSSING_DISTRIBUTIONS = {
    'Bakeri1': lambda: np.random.poisson(8.21),
    'Bakeri2': lambda: np.random.normal(5.66, 2.08),
    'Besat1': lambda: min(10, 2.5 + np.random.lognormal(7.65, 6.42)),
    'Besat2': lambda:  2.5 + np.random.exponential(2.82),
}

# Vehicle type scaling factors
TIME_MODIFIERS = {
    'Bus': {'row_time': 1.5, 'crossing_time': 1.8},
    'Car': {'row_time': 1.0, 'crossing_time': 1.0},
    'Van': {'row_time': 1.2, 'crossing_time': 1.2},
    'Pickup': {'row_time': 1.1, 'crossing_time': 1.1},
}

# Helper Functions
def calculate_row_time(street, row_index):
    return (row_index - 1) * ROW_TIMES[street]

def simulate_vehicle_cycle_time(street, light_times, row_time, crossing_time, direction, sched_value, lane, light_phase, row_index, vehicle_type):
    waiting_time = 0
    encountered_red_light = False
    

    # Apply vehicle type modifiers
    row_time *= TIME_MODIFIERS[vehicle_type]['row_time']
    crossing_time *= TIME_MODIFIERS[vehicle_type]['crossing_time']

    print(f"\n--- Starting Simulation ---")
    print(f"Street: {street}")
    print(f"Lane: {lane}, Row Index: {row_index}")
    print(f"Direction: {direction}")
    print(f"Vehicle Type: {vehicle_type}")
    print(f"Initial Light Phase: {'Green' if light_phase else 'Red'}")
    print(f"Time to Reach Frontline of Intersection: {row_time:.3f} seconds")
    print(f"Crossing Time Required: {crossing_time:.2f} seconds\n")

    while True:
        # Determine action based on lane, direction, and light phase
        if lane == 3 and direction == 'right':
            action = 'cross_intersection'  # Lane 3 vehicles turning right can always cross
        elif lane == 1 and direction == 'left':
            if light_phase:
                action = 'cross_intersection'  # Cross directly if light is green
            else:
                action = 'back_then_cross'  # Wait for red light, then cross
        elif lane == 1 and direction == 'straight':
            if light_phase:
                action = 'move_to_lane2_then_cross'  # Move to lane 2, then cross
            else:
                action = 'back_then_move_to_lane2_then_cross'  # Wait for red light, move to lane 2, then cross
        elif lane == 2 and direction == 'straight':
            if light_phase:
                action = 'cross_intersection'  # Cross directly if light is green
            else:
                action = 'back_then_cross'  # Wait for red light, then cross
        elif lane == 2 and direction == 'left':
            if light_phase:
                action = 'move_to_lane1_then_cross'  # Move to lane 1, then cross
            else:
                action = 'back_then_move_to_lane1_then_cross'  # Wait for red light, move to lane 1, then cross
        else:
            action = 'unspecified'  # Unexpected condition

        # Handle actions and print details
        if action == 'cross_intersection':
            print(f"Action: cross_intersection. Vehicle crosses the intersection.")
            break
        elif action == 'move_to_lane2_then_cross':
            print("Action: move_to_lane2_then_cross. Vehicle moves to lane 2, then crosses the intersection.")
            break
        elif action == 'move_to_lane1_then_cross':
            print("Action: move_to_lane1_then_cross. Vehicle moves to lane 1, then crosses the intersection.")
            break
        elif action == 'back_then_cross':
            # Calculate a random waiting time during red light
            dynamic_red_light_wait = random.uniform(0, light_times['red'][street])
            print(f"Action: back_then_cross. Vehicle waits at the red light for {dynamic_red_light_wait:.2f} seconds, then crosses the intersection.")
            waiting_time += dynamic_red_light_wait
            light_phase = True  # Light turns green after red
        elif action == 'back_then_move_to_lane2_then_cross':
            dynamic_red_light_wait = random.uniform(0, light_times['red'][street])
            print(f"Action: back_then_move_to_lane2_then_cross. Vehicle waits at the red light for {dynamic_red_light_wait:.2f} seconds, then moves to lane 2 and crosses the intersection.")
            waiting_time += dynamic_red_light_wait
            light_phase = True
        elif action == 'back_then_move_to_lane1_then_cross':
            dynamic_red_light_wait = random.uniform(0, light_times['red'][street])
            print(f"Action: back_then_move_to_lane1_then_cross. Vehicle waits at the red light for {dynamic_red_light_wait:.2f} seconds, then moves to lane 1 and crosses the intersection.")
            waiting_time += dynamic_red_light_wait
            light_phase = True
        else:
            print("Action: unspecified. Vehicle remains stationary due to an unexpected condition.")
            break

    # Add the time it took to reach the frontline
    waiting_time += row_time

    # Calculate the total time spent by the vehicle
    total_cycle_time = waiting_time + crossing_time

    # Print the final summary for this vehicle
    print("\n--- Final Summary for Vehicle ---")
    print(f"Total Time Spent by Vehicle: {total_cycle_time:.2f} seconds")
    print(f"  - Time to Reach Frontline: {row_time:.3f} seconds")
    print(f"  - Waiting at Red Light: {waiting_time - row_time:.2f} seconds")
    print(f"  - Crossing Time: {crossing_time:.2f} seconds")
    print(f"------------------------------------------\n")

    return total_cycle_time

def run_simulation(street, light_times, num_iterations, starting_light_phase, max_rows=5):
    cycle_times = []
    light_phase = starting_light_phase
    lane_queues = {'lane1': 0, 'lane2': 0, 'lane3': 0}  # Queue for each lane

    for i in range(num_iterations):
        # Determine vehicle type
        vehicle_type = random.choices(['Bus', 'Car', 'Van', 'Pickup'], weights=VEHICLE_PROBABILITIES[street])[0]
        # Determine direction
        direction = random.choices(['left', 'straight', 'right'], weights=[
            DIRECTION_PROBABILITIES[street]['left'], 
            DIRECTION_PROBABILITIES[street]['straight'], 
            DIRECTION_PROBABILITIES[street]['right']
        ])[0]

        # Assign lane based on direction
        lane = 3 if direction == 'right' else (1 if lane_queues['lane1'] <= lane_queues['lane2'] else 2)
        row_index = lane_queues[f'lane{lane}'] + 1  # Increment row index based on current queue
        lane_queues[f'lane{lane}'] += 1  # Increment the queue size for the lane

        # Calculate times
        row_time = calculate_row_time(street, row_index)
        crossing_time = CROSSING_DISTRIBUTIONS[street]()
        sched_value = random.randint(0, 1)

        print(f"Iteration: {i}, Vehicle: {vehicle_type}")

        # Simulate vehicle cycle time
        cycle_time = simulate_vehicle_cycle_time(
            street, light_times, row_time, crossing_time, direction, sched_value, lane, light_phase, row_index, vehicle_type
        )
        cycle_times.append(cycle_time)

        # Update light phase (toggle green/red)
        light_phase = not light_phase

    return np.mean(cycle_times)

def main():
    scenarios = [3, 4, 7]
    streets = ['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2']
    results = pd.DataFrame(columns=streets)

    total_iterations_per_scenario = 50
    num_iterations_per_street = total_iterations_per_scenario 

    starting_light_phases = {
        'Bakeri1': True,
        'Bakeri2': True,
        'Besat1': False,
        'Besat2': True,
    }

    for scenario in scenarios:
        print(f"\nRunning Scenario {scenario}")
        light_times = LIGHT_TIMES[scenario]
        row = {}
        for street in streets:
            avg_cycle_time = run_simulation(
                street, light_times, num_iterations=num_iterations_per_street, 
                starting_light_phase=starting_light_phases[street], max_rows=5
            )
            row[street] = avg_cycle_time
        results.loc[scenario] = row

    print("\nTraffic Simulation Results")
    print(results)

if __name__ == "__main__":
    main()