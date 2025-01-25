import simpy
import random
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera


#Average row times for each street
ROW_TIMES = {
    'Bakeri1': 2.675,
    'Bakeri2': 2.4666,
    'Besat1': 2.6818,
    'Besat2': 2.3243,
}

#Vehicle type probabilities on each street
VEHICLE_PROBABILITIES = {
    'Bakeri1': [0.06, 0.82, 0.08, 0.04],
    'Bakeri2': [0.04, 0.75, 0.14, 0.07],
    'Besat1': [0.05, 0.80, 0.10, 0.05],
    'Besat2': [0.07, 0.77, 0.09, 0.07],
}

#Direction probabilities on each street
DIRECTION_PROBABILITIES = {
    'Bakeri1': {'left': 0.4, 'straight': 0.5, 'right': 0.1},
    'Bakeri2': {'left': 0.33, 'straight': 0.57, 'right': 0.1},
    'Besat1': {'left': 0.2, 'straight': 0.7, 'right': 0.1},
    'Besat2': {'left': 0.23, 'straight': 0.67, 'right': 0.1},
}


#Crossing time distribution on each street
CROSSING_DISTRIBUTIONS = {
    'Bakeri1': lambda: max(7,np.random.poisson(8.21)),
    'Bakeri2': lambda: max(10,np.random.normal(5.66, 2.08)),
    'Besat1': lambda: min(13, 2.5 + np.random.lognormal(7.65, 6.42)),
    'Besat2': lambda: max(10,2.5 + np.random.exponential(2.82)),
}

#Scale arrival distributions
scale_arrival_bakeri1 = 0.58
scale_arrival_bakeri2 = 0.6
scale_arrival_besat1 = 0.8
scale_arrival_besat2 = 0.8

#Arrival distributions on each street
ARRIVAL_DISTRIBUTIONS = {
    'Bakeri1': lambda: max(0.1, scale_arrival_bakeri1 * (-0.5 + 22 * np.random.beta(0.971, 2.04))),
    'Bakeri2': lambda: max(0.1, scale_arrival_bakeri2 * (-0.5 + 41 * np.random.beta(0.968, 3.44))),
    'Besat1': lambda: max(0.1, scale_arrival_besat1 * (-0.5 + 23 * np.random.beta(0.634, 1.61))),
    'Besat2': lambda: max(0.1, scale_arrival_besat2 * (-0.5 + 24 * np.random.beta(0.963, 1.99))),
}

#Modify crossing and row time for each type of vehicle
TIME_MODIFIERS = {
    'Bus': {'row_time': 1.5, 'crossing_time': 1.8},
    'Car': {'row_time': 1.0, 'crossing_time': 1.0},
    'Van': {'row_time': 1.2, 'crossing_time': 1.2},
    'Pickup': {'row_time': 1.1, 'crossing_time': 1.1},
}

#Green light duration for each street in each scenario
SCENARIO_PHASES = {
    3: [
        {"green": ["Bakeri1"], "duration": 40},    #0-40s
        {"green": ["Bakeri2"], "duration": 30},    #40-70s
        {"green": ["Besat1"], "duration": 25},     #70-95s
        {"green": ["Besat2"], "duration": 25},     #95-120s
    ],
    4: [
        {"green": ["Bakeri1"], "duration": 40},
        {"green": ["Bakeri2"], "duration": 35},
        {"green": ["Besat1"], "duration": 30},
        {"green": ["Besat2"], "duration": 25},
    ],
    7: [
        {"green": ["Bakeri1"], "duration": 30},
        {"green": ["Bakeri2"], "duration": 25},
        {"green": ["Besat1"], "duration": 30},
        {"green": ["Besat2"], "duration": 30},
    ],
}

#Determine which street has green light during simulation
def get_active_phase(current_time, scenario):
    phases = SCENARIO_PHASES[scenario]
    total_cycle = sum(phase["duration"] for phase in phases)
    t_mod = current_time % total_cycle

    #Find which phase contains t_mod
    cumulative = 0
    for phase in phases:
        start = cumulative
        end = cumulative + phase["duration"]
        if start <= t_mod < end:
            return phase
        cumulative = end
    return phases[-1]  #Fallback

#Calculate offsets for each street based on green light 
def calculate_offsets(light_times):
    green_times = light_times['green']
    offsets = {}
    cumulative_time = 0
    for street in ['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2']:
        offsets[street] = cumulative_time
        cumulative_time += green_times[street]
    return offsets

class IntersectionSimulation:
    def __init__(self, env, light_times, arrival_distributions):
        self.env = env
        self.light_times = light_times
        self.arrival_distributions = arrival_distributions
        self.queues = {street: {1: [], 2: [], 3: []} for street in arrival_distributions}
        self.light_state = {
            'Bakeri1': False,
            'Bakeri2': False,
            'Besat1': False,
            'Besat2': False
        }
        self.results = {street: [] for street in arrival_distributions}
        self.vehicle_count = 0

        self.env.process(self.traffic_light_cycle())
        for street in arrival_distributions:
            self.env.process(self.vehicle_arrival(street))

	#Controls the traffic light system
    def traffic_light_cycle(self):
        #Define order of streets for the light cycle
        street_order = ['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2']
        cycle_order = [(street, self.light_times['green'][street]) for street in street_order]
        total_cycle_time = sum(self.light_times['green'].values())  #Total cycle time for all streets

        while True:
            for street, green_time in cycle_order:
                #Ensure all lights are red before turning any green
                for other_street in self.light_state:
                    self.light_state[other_street] = False

                
                print(f"At {self.env.now:.2f}, Turning GREEN for {street}. All others are RED.")

                #Turn on green light for the current street
                self.light_state[street] = True

               
                print(f"At {self.env.now:.2f}, Green Light ON for {street}. Light States: {self.light_state}")

                #Keep green light on for the specified duration
                yield self.env.timeout(green_time)

                
                print(f"At {self.env.now:.2f}, End of Green Light for {street}. All lights RED.")

	#Setup vehicle arriving at intersection
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

	#Crossing intersection
    def vehicle_cross(self, street, vehicle_id, direction, queue_num, vehicle_type):
        arrival_time = self.env.now
        queue_position = len(self.queues[street][queue_num])  #Vehicle's position in queue
        row_index = queue_position + 1 
        row_wait_time = (row_index - 1) * ROW_TIMES[street] * TIME_MODIFIERS[vehicle_type]['row_time']

        #Add vehicle to queue
        self.queues[street][queue_num].append(vehicle_id)
        yield self.env.timeout(row_wait_time)  #Wait time to move to the first row

        #Wait for green light if not turning right
        red_light_wait_time = 0
        if direction != 'right':
            while not self.light_state[street]:
                print(f"At {self.env.now:.2f}, Vehicle {vehicle_id} on {street} encountered RED light")
                yield self.env.timeout(1)
                red_light_wait_time += 1
        else:
            if not self.light_state[street]:  #If red light but turning right
                print(f"At {self.env.now:.2f}, Vehicle {vehicle_id} on {street} encountered RED light but is TURNING RIGHT")


        print(f"At {self.env.now:.2f}, Vehicle {vehicle_id} on {street} encountered GREEN light")
        #Calculate crossing time 
        crossing_time = CROSSING_DISTRIBUTIONS[street]() * TIME_MODIFIERS[vehicle_type]['crossing_time']
        yield self.env.timeout(crossing_time)

        #Remove vehicle from queue
        self.queues[street][queue_num].remove(vehicle_id)
        total_time_spent = self.env.now - arrival_time



        #Record total time 
        self.results[street].append(total_time_spent)

      
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

#Run simulation
def run_simulation(scenarios):
    results = pd.DataFrame(columns=['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2','Average'])
    for scenario in scenarios:
        env = simpy.Environment()
        light_times = LIGHT_TIMES[scenario]
        sim = IntersectionSimulation(env, light_times, ARRIVAL_DISTRIBUTIONS)
        sim.scenario = scenario
        env.run(until=3000)

        
        print("\n--- Results Content ---")
        for street, times in sim.results.items():
            print(f"{street}: {times}")

        avg_results = sim.get_results()
        avg_results['Average'] = np.mean(list(avg_results.values()))
        avg_results['Bakeri Average'] = np.mean([avg_results['Bakeri1'], avg_results['Bakeri2']])
        results.loc[scenario] = avg_results

       
        print("\n--- Manual Averages ---")
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

#Perform normality test and build confidence interval
def analyze_results(data):
        jb_stat, jb_pvalue = jarque_bera(data)
        print(f"\nJarque-Bera Test Results:")
        print(f"Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")

        is_normal = jb_pvalue > 0.05
        if is_normal:
            print("The data PASSES the normality test (p-value > 0.05).")
        else:
            print("The data DOES NOT PASS the normality test (p-value <= 0.05).")
        
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        n = len(data)
        confidence_level = 0.95
        z_value = 1.96  
        margin_of_error = z_value * (std_dev / np.sqrt(n))

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        print("\nConfidence Interval (95%):")
        print(f"Mean: {mean:.4f}, Std Dev: {std_dev:.4f}")
        print(f"Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}")
        
        return {"mean": mean, "std_dev": std_dev, "lower_bound": lower_bound, "upper_bound": upper_bound}

#Run simulation 50 times and perform normality test and build confidence interval
def run_and_analyze(scenario, num_runs):
    scenario_averages = []
    cumulative_results = pd.DataFrame(columns=['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2', 'Average'])

    for _ in range(num_runs):
            results = run_simulation([scenario])  
            scenario_average = results.loc[scenario, 'Average']  
            scenario_averages.append(scenario_average)  

            cumulative_results = pd.concat([cumulative_results, results.loc[[scenario]]], ignore_index=True)
        
    final_averages = cumulative_results.mean()

    print(f"\nTraffic Simulation Results for {num_runs} iterations")
    print(final_averages.to_frame(name='Average').T)
    
    analyze_results(scenario_averages)

num_runs = 50
scenario = 3
run_and_analyze(scenario,num_runs ) 

scenario = 4
run_and_analyze(scenario,num_runs ) 

scenario = 7
run_and_analyze(scenario,num_runs ) 