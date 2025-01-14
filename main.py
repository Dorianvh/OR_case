import simpy
import random
import statistics
import numpy as np

##############################################################################
# Arrival and Crossing Distributions
##############################################################################
def arrival_time_bakeri1():
    """-0.5 + 22 * Beta(0.971, 2.04)"""
    val = -0.5 + 22 * np.random.beta(a=0.971, b=2.04)
    return max(val, 0.0)  # Ensure nonnegative

def arrival_time_bakeri2():
    """-0.5 + 41 * Beta(0.968, 3.44)"""
    val = -0.5 + 41 * np.random.beta(a=0.968, b=3.44)
    return max(val, 0.0)

def arrival_time_besat1():
    """-0.5 + 23 * Beta(0.634, 1.61)"""
    val = -0.5 + 23 * np.random.beta(a=0.634, b=1.61)
    return max(val, 0.0)

def arrival_time_besat2():
    """-0.5 + 24 * Beta(0.963, 1.99)"""
    val = -0.5 + 24 * np.random.beta(a=0.963, b=1.99)
    return max(val, 0.0)

def crossing_time_bakeri1():
    """Poisson(8.21)"""
    return np.random.poisson(lam=8.21)

def crossing_time_bakeri2():
    """Normal(5.66, 2.08)"""
    val = np.random.normal(loc=5.66, scale=2.08)
    return max(val, 0.0)  # Clip negatives

def crossing_time_besat1():
    """2.5 + Lognormal(7.65, 6.42)"""
    val = 2.5 + np.random.lognormal(mean=7.65, sigma=6.42)
    return min(val, 10)  # Clip negatives

def crossing_time_besat2():
    """2.5 + Erlang(2.82, 3) => Gamma(k=3, scale=2.82)"""
    return 2.5 + np.random.gamma(shape=3, scale=2.82)


##############################################################################
# Vehicle / Traffic Light Logic
##############################################################################
def vehicle_generator(env, street_name, arrival_func, cross_func,
                      light_state, results, queue_counts, row_times):
    """
    Generates vehicles for 'street_name'. Each new vehicle increases
    that street's queue length and then is passed to vehicle() process.
    """
    i = 0
    while True:
        # Inter-arrival time for next vehicle
        yield env.timeout(arrival_func())

        i += 1
        # Increase the queue count for this street
        queue_counts[street_name] += 1

        vid = f"{street_name}-{i}"
        env.process(vehicle(env, vid, street_name, cross_func,
                            light_state, results, queue_counts, row_times))

def vehicle(env, vid, street_name, cross_func,
            light_state, results, queue_counts, row_times):
    """
    The vehicle logic, including extra row-based waiting time
    before reaching the 'front row'.
    """
    # We assume queue_counts[street_name] was already incremented in vehicle_generator.
    position_in_queue = queue_counts[street_name]

    # Table 2: Row-based waiting time, e.g. Bakeri1 = 2.675 seconds per row
    # (i-1) * row_time
    delay_to_front = (position_in_queue - 1) * row_times[street_name]

    t0 = env.now
    # Wait the row-based delay before even being at the stop line
    yield env.timeout(delay_to_front)

    # Decide turning right or going straight/left
    direction = random.choice(["left", "straight", "right"])

    if direction == 'right':
        # Right turn always allowed, no blocking modeled => half crossing time
        yield env.timeout(0.5 * cross_func())
    else:
        # Must wait for green
        while True:
            if street_name == 'Bakeri1' and light_state['bakeri1']:
                break
            elif street_name == 'Bakeri2' and light_state['bakeri2']:
                break
            elif street_name == 'Besat1' and light_state['besat1']:
                break
            elif street_name == 'Besat2' and light_state['besat2']:
                break
            # Re-check every 0.5s
            yield env.timeout(0.5)
        # Now cross
        yield env.timeout(cross_func())

    # Record total time
    total_time = env.now - t0
    results.append(total_time)

    # This vehicle is done; reduce the queue count
    queue_counts[street_name] -= 1

def traffic_light_controller_4stroke(env, light_state,
                                     green_bakeri1,
                                     green_bakeri2,
                                     green_besat1,
                                     green_besat2,
                                     yellow=0):
    """
    A 4-stroke traffic light controller with optional 'yellow' times.
    """
    while True:
        # PHASE 1
        light_state['bakeri1'] = True
        light_state['bakeri2'] = False
        light_state['besat1']  = False
        light_state['besat2']  = False
        yield env.timeout(green_bakeri1)
        if yellow > 0:
            light_state['bakeri1'] = False
            yield env.timeout(yellow)

        # PHASE 2
        light_state['bakeri1'] = False
        light_state['bakeri2'] = True
        light_state['besat1']  = False
        light_state['besat2']  = False
        yield env.timeout(green_bakeri2)
        if yellow > 0:
            light_state['bakeri2'] = False
            yield env.timeout(yellow)

        # PHASE 3
        light_state['bakeri1'] = False
        light_state['bakeri2'] = False
        light_state['besat1']  = True
        light_state['besat2']  = False
        yield env.timeout(green_besat1)
        if yellow > 0:
            light_state['besat1'] = False
            yield env.timeout(yellow)

        # PHASE 4
        light_state['bakeri1'] = False
        light_state['bakeri2'] = False
        light_state['besat1']  = False
        light_state['besat2']  = True
        yield env.timeout(green_besat2)
        if yellow > 0:
            light_state['besat2'] = False
            yield env.timeout(yellow)

def run_scenario(scenario_id, green_times, sim_duration=3600, seed=0):
    """
    Runs the 4-stroke intersection simulation with given green times,
    including row-based waiting from Table 2.
    """
    random.seed(seed)
    np.random.seed(seed)  # If you also want deterministic numpy draws

    env = simpy.Environment()

    # Track which street is green
    light_state = {
        'bakeri1': False,
        'bakeri2': False,
        'besat1':  False,
        'besat2':  False
    }

    # Unpack green times
    g_b1, g_b2, g_bs1, g_bs2 = green_times

    # Start the 4-stroke traffic light
    env.process(
        traffic_light_controller_4stroke(env, light_state,
                                         green_bakeri1 = g_b1,
                                         green_bakeri2 = g_b2,
                                         green_besat1  = g_bs1,
                                         green_besat2  = g_bs2,
                                         yellow=0)
    )

    # Table 2 row-based times (seconds to move from row i to row i-1)
    row_times = {
        'Bakeri1': 2.675,
        'Bakeri2': 2.4666,
        'Besat1':  2.6818,
        'Besat2':  2.3243
    }

    # We'll keep a simple queue count per street
    queue_counts = {
        'Bakeri1': 0,
        'Bakeri2': 0,
        'Besat1':  0,
        'Besat2':  0
    }

    # Lists to hold per-vehicle times
    results_bakeri1 = []
    results_bakeri2 = []
    results_besat1  = []
    results_besat2  = []

    # Start vehicle generators
    env.process(vehicle_generator(env, 'Bakeri1',
                                  arrival_time_bakeri1,
                                  crossing_time_bakeri1,
                                  light_state, results_bakeri1,
                                  queue_counts, row_times))
    env.process(vehicle_generator(env, 'Bakeri2',
                                  arrival_time_bakeri2,
                                  crossing_time_bakeri2,
                                  light_state, results_bakeri2,
                                  queue_counts, row_times))
    env.process(vehicle_generator(env, 'Besat1',
                                  arrival_time_besat1,
                                  crossing_time_besat1,
                                  light_state, results_besat1,
                                  queue_counts, row_times))
    env.process(vehicle_generator(env, 'Besat2',
                                  arrival_time_besat2,
                                  crossing_time_besat2,
                                  light_state, results_besat2,
                                  queue_counts, row_times))

    # Run the simulation
    env.run(until=sim_duration)

    # Compute average times
    def avg_or_zero(lst):
        return statistics.mean(lst) if lst else 0.0

    return (avg_or_zero(results_bakeri1),
            avg_or_zero(results_bakeri2),
            avg_or_zero(results_besat1),
            avg_or_zero(results_besat2))

##############################################################################
# Example: Running Multiple Scenarios
##############################################################################
if __name__ == '__main__':
    scenarios = {
        3: (40, 30, 25, 25),
        4: (40, 35, 30, 25),
        7: (30, 25, 30, 30),
    }

    for sc_id, greens in scenarios.items():
        (r_b1, r_b2, r_s1, r_s2) = run_scenario(sc_id, greens,
                                               sim_duration=3600,
                                               seed=42)
        overall_avg = statistics.mean([r_b1, r_b2, r_s1, r_s2])
        print(f"Scenario {sc_id}:")
        print(f"  Bakeri1 avg time = {r_b1:.2f} s")
        print(f"  Bakeri2 avg time = {r_b2:.2f} s")
        print(f"  Besat1  avg time = {r_s1:.2f} s")
        print(f"  Besat2  avg time = {r_s2:.2f} s")
        print(f"  Overall avg time = {overall_avg:.2f} s\n")
