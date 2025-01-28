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

    def spsa(self):
        print()

# --------------------------------------------------
# 3) TKINTER CANVAS GUI
# --------------------------------------------------

class IntersectionCanvasGUI:
    """
    A Canvas-based GUI that:
      - Draws a stylized 4-way intersection with 3 lanes each side
      - Shows color-coded traffic lights
      - Displays simulation time
      - Draws vehicle "dots" in each lane to indicate queue length
      - Labels each street by name
    """

    def __init__(self, master, light_times, arrival_distributions, run_time=200):
        self.master = master
        self.master.title("Intersection Simulation with Vehicle Queues")

        # SimPy environment
        self.env = simpy.Environment()
        self.sim = IntersectionSimulation(self.env, light_times, arrival_distributions)
        self.run_time = run_time

        # Top frame: time display
        top_frame = ttk.Frame(master)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        self.time_label = ttk.Label(top_frame, text="Time: 0.0", font=("TkDefaultFont", 12, "bold"))
        self.time_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Canvas for intersection
        self.canvas = tk.Canvas(master, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Draw static geometry (roads, lane dividers, street names)
        self.draw_static_intersection()
        self.draw_street_names()

        # Place traffic lights (circles on canvas)
        self.light_shapes = {}
        self.draw_lights()

        # We'll store references to vehicle "dots" so we can remove/redraw
        self.vehicle_shapes = []

        # Begin simulation stepping
        self.step_simulation()

    # ---------- Drawing the roads and text ----------

    def draw_static_intersection(self):
        """Draw wide gray roads for Bakeri1 (top), Bakeri2 (bottom), Besat1 (left), Besat2 (right)."""
        # Intersection center ~ (300,300), each road about 100 px wide

        # Bakeri1 top road
        self.canvas.create_rectangle(250, 0, 350, 300, fill="lightgray")
        self.draw_lane_dividers_vertical(x_center=300, y_start=0, y_end=300)

        # Bakeri2 bottom road
        self.canvas.create_rectangle(250, 300, 350, 600, fill="lightgray")
        self.draw_lane_dividers_vertical(x_center=300, y_start=300, y_end=600)

        # Besat1 left road
        self.canvas.create_rectangle(0, 250, 300, 350, fill="lightgray")
        self.draw_lane_dividers_horizontal(y_center=300, x_start=0, x_end=300)

        # Besat2 right road
        self.canvas.create_rectangle(300, 250, 600, 350, fill="lightgray")
        self.draw_lane_dividers_horizontal(y_center=300, x_start=300, x_end=600)

    def draw_lane_dividers_vertical(self, x_center, y_start, y_end):
        """
        Draw dashed white lane dividers for a vertical road.
        We'll make multiple lines offset from center.
        """
        offsets = [10, 20, 30, 40, 50]
        for off in offsets:
            # Left side
            x = x_center - off
            self.canvas.create_line(x, y_start, x, y_end, fill="white", dash=(5,2))
            # Right side
            x2 = x_center + off
            self.canvas.create_line(x2, y_start, x2, y_end, fill="white", dash=(5,2))

    def draw_lane_dividers_horizontal(self, y_center, x_start, x_end):
        """
        Draw dashed white lane dividers for a horizontal road.
        """
        offsets = [10, 20, 30, 40, 50]
        for off in offsets:
            # Up side
            y = y_center - off
            self.canvas.create_line(x_start, y, x_end, y, fill="white", dash=(5,2))
            # Down side
            y2 = y_center + off
            self.canvas.create_line(x_start, y2, x_end, y2, fill="white", dash=(5,2))

    def draw_street_names(self):
        """
        Label each street near the road approach.
        """
        # Bakeri1 => top
        self.canvas.create_text(300, 30, text="Bakeri1", fill="black", font=("Arial", 14, "bold"))
        # Bakeri2 => bottom
        self.canvas.create_text(300, 570, text="Bakeri2", fill="black", font=("Arial", 14, "bold"))
        # Besat1 => left (rotate text 90 deg if desired)
        self.canvas.create_text(30, 300, text="Besat1", fill="black", font=("Arial", 14, "bold"), angle=90)
        # Besat2 => right
        self.canvas.create_text(570, 300, text="Besat2", fill="black", font=("Arial", 14, "bold"), angle=-90)

    # ---------- Drawing traffic lights and vehicles ----------

    def draw_lights(self):
        """
        Draw small circles indicating traffic lights for each street.
        We'll place them near the intersection.
        """
        r = 10
        # Bakeri1 => top
        top_light = self.canvas.create_oval(290, 40, 310, 60, fill="red")
        self.light_shapes['Bakeri1'] = top_light

        # Bakeri2 => bottom
        bottom_light = self.canvas.create_oval(290, 540, 310, 560, fill="red")
        self.light_shapes['Bakeri2'] = bottom_light

        # Besat1 => left
        left_light = self.canvas.create_oval(40, 290, 60, 310, fill="red")
        self.light_shapes['Besat1'] = left_light

        # Besat2 => right
        right_light = self.canvas.create_oval(540, 290, 560, 310, fill="red")
        self.light_shapes['Besat2'] = right_light

    # For placing vehicle "dots" in each lane, define offsets:
    # Each dictionary entry => (x_start, y_start, dx, dy)
    # so we can position each queued vehicle by index i.
    # This is just an approximate layout for demonstration.
    lane_positions = {
        # Bakeri1 => top approach, going down. 3 lanes side-by-side
        ('Bakeri1', 1): (280, 100, 0, -10),  # left lane, stack upward behind
        ('Bakeri1', 2): (300, 100, 0, -10),  # middle lane
        ('Bakeri1', 3): (320, 100, 0, -10),  # right lane

        # Bakeri2 => bottom approach, going up
        ('Bakeri2', 1): (280, 500, 0, 10),   # left lane, stack downward
        ('Bakeri2', 2): (300, 500, 0, 10),
        ('Bakeri2', 3): (320, 500, 0, 10),

        # Besat1 => left approach, going right
        ('Besat1', 1): (100, 280, -10, 0),
        ('Besat1', 2): (100, 300, -10, 0),
        ('Besat1', 3): (100, 320, -10, 0),

        # Besat2 => right approach, going left
        ('Besat2', 1): (500, 280, 10, 0),
        ('Besat2', 2): (500, 300, 10, 0),
        ('Besat2', 3): (500, 320, 10, 0),
    }

    # --------------------------------------------------
    # 4) SIMULATION LOOP
    # --------------------------------------------------

    def step_simulation(self):
        """Step the simulation in small increments, update GUI each time."""
        if self.env.now >= self.run_time:
            return
        try:
            self.env.step()  # one event
        except simpy.core.EmptySchedule:
            return

        self.update_gui()
        # Update every 50 ms in real time
        self.master.after(10, self.step_simulation)

    def update_gui(self):
        """Redraw time display, traffic lights, and vehicle queues."""
        current_time = self.env.now
        self.time_label.config(text=f"Time: {current_time:.1f}")

        # Update light colors
        for street, is_green in self.sim.light_state.items():
            shape_id = self.light_shapes[street]
            self.canvas.itemconfig(shape_id, fill=("green" if is_green else "red"))

        # Clear old vehicle shapes
        for shape_id in self.vehicle_shapes:
            self.canvas.delete(shape_id)
        self.vehicle_shapes.clear()

        # Draw a small square for each vehicle in each queue
        # The queue order is from front to back, so index i=0 is front.
        # We'll just space them out along the approach with an offset.
        for street, lane_dict in self.sim.queues.items():
            for lane_num, queue_list in lane_dict.items():
                # (x_start, y_start, dx, dy) for lane
                if (street, lane_num) not in self.lane_positions:
                    continue
                x0, y0, dx, dy = self.lane_positions[(street, lane_num)]

                # For each vehicle in the queue, place a small 6x6 box
                for i, vehicle_id in enumerate(queue_list):
                    # offset from start pos
                    xx = x0 + i * dx
                    yy = y0 + i * dy
                    size = 3  # half-size => 6x6 box
                    shape_id = self.canvas.create_rectangle(
                        xx - size, yy - size, xx + size, yy + size,
                        fill="blue", outline=""
                    )
                    self.vehicle_shapes.append(shape_id)

# --------------------------------------------------
# 5) SPSA OPTIMIZATION
# --------------------------------------------------

class SPSAOptimizer:
    def __init__(self, sim_class, initial_theta, a=0.1, c=0.1, alpha=0, gamma=0, max_iter=1000):
        self.sim_class = sim_class
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

            # Update theta with a lower bound of 1
            self.theta = np.maximum(1, self.theta - ak[k] * gk)

            # Store the current theta and loss
            self.theta_history.append(self.theta.copy())
            self.loss_history.append(self.evaluate(self.theta))

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
        sim = self.sim_class(env, LIGHT_TIMES[3], ARRIVAL_DISTRIBUTIONS, thresholds)
        env.run(until=3600)
        average_time = sim.get_average_time()

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
    initial_theta = [15, 15, 15, 15]  # Initial guess for thresholds for each street
    optimizer = SPSAOptimizer(IntersectionSimulation, initial_theta)
    optimal_theta = optimizer.optimize()
    print(f"Optimal THRESHOLDS: {optimal_theta}")
    optimizer.plot_results()

def main2():
    root = tk.Tk()
    scenario = 3  # we have only 3 in LIGHT_TIMES as an example
    app = IntersectionCanvasGUI(root, LIGHT_TIMES[scenario], ARRIVAL_DISTRIBUTIONS, run_time=1000)
    root.mainloop()


    print("Simulation ended.")
    results = app.sim.get_results()
    for street, avg_time in results.items():
        print(f"{street} average time = {avg_time:.2f} sec")

    print(f"Total average time = {app.sim.get_average_time():.2f} sec")

if __name__ == "__main__":
    main()
