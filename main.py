import simpy
import random
import statistics
import numpy as np

import tkinter as tk
from tkinter import ttk

##############################################################################
# 1) GLOBAL STRUCTURES FOR COLLECTING SNAPSHOTS
##############################################################################
animation_data = []  # Will hold a list of snapshots

# For demonstration, track how many cars in queue & crossing
queue_counts = {
    'Bakeri1': 0,
    'Bakeri2': 0,
    'Besat1':  0,
    'Besat2':  0
}
current_crossings = {
    'Bakeri1': 0,
    'Bakeri2': 0,
    'Besat1':  0,
    'Besat2':  0
}

##############################################################################
# 2) DISTRIBUTIONS FOR ARRIVAL AND CROSSING TIMES
##############################################################################
def arrival_time_bakeri1():
    val = -0.5 + 22 * np.random.beta(a=0.971, b=2.04)
    return max(val, 0.0)

def arrival_time_bakeri2():
    val = -0.5 + 41 * np.random.beta(a=0.968, b=3.44)
    return max(val, 0.0)

def arrival_time_besat1():
    val = -0.5 + 23 * np.random.beta(a=0.634, b=1.61)
    return max(val, 0.0)

def arrival_time_besat2():
    val = -0.5 + 24 * np.random.beta(a=0.963, b=1.99)
    return max(val, 0.0)

def crossing_time_bakeri1():
    return np.random.poisson(lam=8.21)

def crossing_time_bakeri2():
    # Normal can be negative; clip it
    val = np.random.normal(loc=5.66, scale=2.08)
    return max(val, 0.0)

def crossing_time_besat1():
    return 2.5 + np.random.lognormal(mean=7.65, sigma=6.42)

def crossing_time_besat2():
    return 2.5 + np.random.gamma(shape=3, scale=2.82)

##############################################################################
# ROW-BASED DELAYS (TABLE 2)
##############################################################################
row_times = {
    'Bakeri1': 2.675,
    'Bakeri2': 2.4666,
    'Besat1':  2.6818,
    'Besat2':  2.3243
}

##############################################################################
# 3) SIMPY PROCESSES
##############################################################################
def vehicle_generator(env, street_name, arrival_func, cross_func,
                      light_state, results):
    i = 0
    while True:
        yield env.timeout(arrival_func())
        i += 1
        queue_counts[street_name] += 1
        vid = f"{street_name}-{i}"
        env.process(vehicle(env, vid, street_name, cross_func,
                            light_state, results))

def vehicle(env, vid, street_name, cross_func, light_state, results):
    # row-based waiting
    position_in_queue = queue_counts[street_name]
    delay_to_front = (position_in_queue - 1) * row_times[street_name]

    t0 = env.now
    yield env.timeout(delay_to_front)

    direction = random.choice(["left","straight","right"])
    if direction == 'right':
        current_crossings[street_name] += 1
        yield env.timeout(0.5 * cross_func())
        current_crossings[street_name] -= 1
    else:
        while True:
            if street_name == 'Bakeri1' and light_state['bakeri1']:
                break
            elif street_name == 'Bakeri2' and light_state['bakeri2']:
                break
            elif street_name == 'Besat1' and light_state['besat1']:
                break
            elif street_name == 'Besat2' and light_state['besat2']:
                break
            yield env.timeout(0.5)
        current_crossings[street_name] += 1
        yield env.timeout(cross_func())
        current_crossings[street_name] -= 1

    total_time = env.now - t0
    results.append(total_time)
    queue_counts[street_name] -= 1

def traffic_light_controller_4stroke(env, light_state,
                                     green_b1, green_b2,
                                     green_s1, green_s2,
                                     yellow=0):
    while True:
        # PHASE 1
        light_state['bakeri1'] = True
        light_state['bakeri2'] = False
        light_state['besat1']  = False
        light_state['besat2']  = False
        yield env.timeout(green_b1)
        if yellow > 0:
            light_state['bakeri1'] = False
            yield env.timeout(yellow)

        # PHASE 2
        light_state['bakeri1'] = False
        light_state['bakeri2'] = True
        light_state['besat1']  = False
        light_state['besat2']  = False
        yield env.timeout(green_b2)
        if yellow > 0:
            light_state['bakeri2'] = False
            yield env.timeout(yellow)

        # PHASE 3
        light_state['bakeri1'] = False
        light_state['bakeri2'] = False
        light_state['besat1']  = True
        light_state['besat2']  = False
        yield env.timeout(green_s1)
        if yellow > 0:
            light_state['besat1'] = False
            yield env.timeout(yellow)

        # PHASE 4
        light_state['bakeri1'] = False
        light_state['bakeri2'] = False
        light_state['besat1']  = False
        light_state['besat2']  = True
        yield env.timeout(green_s2)
        if yellow > 0:
            light_state['besat2'] = False
            yield env.timeout(yellow)

##############################################################################
# 4) CAPTURE “SNAPSHOTS” EVERY 1s FOR LATER PLAYBACK
##############################################################################
def record_state(env, light_state, interval=1.0):
    """Every 'interval' seconds, append a snapshot to 'animation_data'."""
    while True:
        snap = {
            'time':    env.now,
            'lights':  dict(light_state),
            'queues':  dict(queue_counts),
            'crossing':dict(current_crossings)
        }
        animation_data.append(snap)
        yield env.timeout(interval)

##############################################################################
# 5) RUN THE SIMULATION OFFLINE, STORE SNAPSHOTS
##############################################################################
def run_scenario(green_times, sim_duration=1200, seed=42):
    animation_data.clear()

    # reset global counters for each run
    for k in queue_counts:
        queue_counts[k] = 0
    for k in current_crossings:
        current_crossings[k] = 0

    random.seed(seed)
    np.random.seed(seed)

    env = simpy.Environment()
    light_state = {
        'bakeri1': False,
        'bakeri2': False,
        'besat1':  False,
        'besat2':  False
    }

    g_b1, g_b2, g_s1, g_s2 = green_times
    env.process(traffic_light_controller_4stroke(env, light_state,
                                                 g_b1, g_b2, g_s1, g_s2))
    results_bakeri1 = []
    results_bakeri2 = []
    results_besat1  = []
    results_besat2  = []

    env.process(vehicle_generator(env, 'Bakeri1', arrival_time_bakeri1,
                                  crossing_time_bakeri1, light_state, results_bakeri1))
    env.process(vehicle_generator(env, 'Bakeri2', arrival_time_bakeri2,
                                  crossing_time_bakeri2, light_state, results_bakeri2))
    env.process(vehicle_generator(env, 'Besat1', arrival_time_besat1,
                                  crossing_time_besat1, light_state, results_besat1))
    env.process(vehicle_generator(env, 'Besat2', arrival_time_besat2,
                                  crossing_time_besat2, light_state, results_besat2))

    # Record snapshots every 1 second
    env.process(record_state(env, light_state, interval=1.0))

    env.run(until=sim_duration)

    def avg_or_zero(lst):
        return statistics.mean(lst) if lst else 0.0

    return (avg_or_zero(results_bakeri1),
            avg_or_zero(results_bakeri2),
            avg_or_zero(results_besat1),
            avg_or_zero(results_besat2))

##############################################################################
# 6) TKINTER GUI FOR “PLAYBACK” OF THE STORED SNAPSHOTS
##############################################################################
class TrafficSimGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Intersection Simulation")

        self.frame = ttk.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # We have 4 streets: show lights, queue length, crossing
        self.lights_labels   = {}
        self.queues_labels   = {}
        self.crossing_labels = {}

        # Streets in a consistent order
        self.streets = ['Bakeri1', 'Bakeri2', 'Besat1', 'Besat2']

        row_idx = 0
        header = ttk.Label(self.frame, text="Traffic Light Simulation", font=("Arial", 14, "bold"))
        header.grid(row=row_idx, column=0, columnspan=3, pady=(0,10))
        row_idx += 1

        # Table header
        ttk.Label(self.frame, text="Street",  width=10, anchor="center").grid(row=row_idx, column=0, padx=5)
        ttk.Label(self.frame, text="Light",   width=10, anchor="center").grid(row=row_idx, column=1, padx=5)
        ttk.Label(self.frame, text="Queue",   width=10, anchor="center").grid(row=row_idx, column=2, padx=5)
        ttk.Label(self.frame, text="Crossing",width=10, anchor="center").grid(row=row_idx, column=3, padx=5)
        row_idx += 1

        for st in self.streets:
            ttk.Label(self.frame, text=st, width=10).grid(row=row_idx, column=0)
            lbl_light = ttk.Label(self.frame, text="RED", width=10, anchor="center", foreground="red")
            lbl_light.grid(row=row_idx, column=1)
            lbl_queue = ttk.Label(self.frame, text="0", width=10, anchor="center")
            lbl_queue.grid(row=row_idx, column=2)
            lbl_cross = ttk.Label(self.frame, text="0", width=10, anchor="center")
            lbl_cross.grid(row=row_idx, column=3)

            self.lights_labels[st]   = lbl_light
            self.queues_labels[st]   = lbl_queue
            self.crossing_labels[st] = lbl_cross

            row_idx += 1

        # Time label
        self.time_label = ttk.Label(self.frame, text="Time = 0.0 s", font=("Arial", 12))
        self.time_label.grid(row=row_idx, column=0, columnspan=4, pady=(10,0))

        # Start with frame=0
        self.current_frame = 0

    def update_display(self, snapshot):
        """Use 'snapshot' to update the labels in the GUI."""
        sim_time = snapshot['time']
        lights   = snapshot['lights']
        queues   = snapshot['queues']
        crossing = snapshot['crossing']

        self.time_label.config(text=f"Time = {sim_time:.1f} s")

        for st in self.streets:
            # Light color
            if lights[st.lower()]:  # dictionary keys are 'bakeri1' in lowercase
                self.lights_labels[st].config(text="GREEN", foreground="green")
            else:
                self.lights_labels[st].config(text="RED", foreground="red")

            # Queue
            qval = queues[st]
            self.queues_labels[st].config(text=str(qval))

            # Crossing
            cval = crossing[st]
            self.crossing_labels[st].config(text=str(cval))

    def play_snapshots(self):
        """Called repeatedly with 'after' to advance frames slowly."""
        if self.current_frame >= len(animation_data):
            # Done with playback
            return

        snap = animation_data[self.current_frame]
        self.update_display(snap)
        self.current_frame += 1

        # Wait 500 ms (0.5 s) before showing the next frame
        self.master.after(500, self.play_snapshots)

##############################################################################
# 7) MAIN DEMO
##############################################################################
if __name__ == '__main__':
    # 1) Run the simulation "offline" to gather snapshots
    # Example green times: (30, 25, 30, 30)
    r_b1, r_b2, r_s1, r_s2 = run_scenario((30, 25, 30, 30), sim_duration=60, seed=42)
    overall_avg = statistics.mean([r_b1, r_b2, r_s1, r_s2])
    print("Simulation finished.")
    print(f"  Bakeri1 avg time = {r_b1:.2f}")
    print(f"  Bakeri2 avg time = {r_b2:.2f}")
    print(f"  Besat1  avg time = {r_s1:.2f}")
    print(f"  Besat2  avg time = {r_s2:.2f}")
    print(f"  Overall avg time = {overall_avg:.2f}\n")

    # 2) Then create a Tkinter GUI to “play back” the stored snapshots slowly
    root = tk.Tk()
    gui = TrafficSimGUI(root)

    # Start the animation
    gui.play_snapshots()

    root.mainloop()
