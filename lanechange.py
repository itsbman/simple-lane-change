import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as an
from matplotlib.animation import FuncAnimation


def equalize_wp_delta(waypoints, delta_wp=0.5):
    """
    Make path points equidistant
    """
    dist = np.zeros(waypoints.shape[0])
    dist_vector = np.sum((waypoints[1:]
                          - waypoints[:-1])**2, axis=1)**0.5
    dist[1:] = np.cumsum(dist_vector)

    xa, xb = np.zeros((dist.size, 2)), np.zeros((dist.size, 2))
    for j in range(dist.size - 1):
        xa[j, :] = np.matmul(np.linalg.inv([[dist[j], 1], [dist[j + 1], 1]]),
                             [waypoints[j, 0], waypoints[j+1, 0]])
        xb[j, :] = np.matmul(np.linalg.inv([[dist[j], 1], [dist[j + 1], 1]]),
                             [waypoints[j, 1], waypoints[j+1, 1]])

    d = np.arange(dist[0], dist[-1], delta_wp)
    new_wp = np.zeros((d.size, 2))

    k = 0
    for i, di in enumerate(d):
        while di > dist[k]:
            k += 1
        x = xa[k, 0]*di + xa[k, 1]
        y = xb[k, 0]*di + xb[k, 1]
        new_wp[i, :] = [x, y]

    return new_wp[:-1]


def get_waypoint_pos(pos: list, waypoints: numpy.ndarray):
    """
    Find closest waypoint to position
    """
    dist = np.sum((pos - waypoints)**2, axis=1)
    wp_id = np.argmin(dist)
    return waypoints[wp_id], wp_id


def lane_change_waypoint(l1: numpy.ndarray,  # current lane
                         l2: numpy.ndarray,  # target lane
                         start_pos: list,  # position of vehicle
                         forward_idx: int = 26,  # length of path in terms of index
                         delay_idx: int = 4):  # index delay before lane change starts

    init_wp, init_wp_id = get_waypoint_pos(start_pos, l1)

    start_wp_id = delay_idx + init_wp_id

    target_pos, target_id = get_waypoint_pos(l1[start_wp_id + forward_idx, :], l2)

    r_temp = target_pos - l1[start_wp_id + forward_idx]
    r_end = np.sum(r_temp ** 2) ** 0.5
    r_temp /= np.hypot(r_temp[0], r_temp[1])

    s_vector = np.zeros(forward_idx)
    dist_vector = np.sum((l1[start_wp_id+1:start_wp_id+forward_idx, :]
                          - l1[start_wp_id:start_wp_id+forward_idx-1, :])**2, axis=1)**0.5
    s_vector[1:] = np.cumsum(dist_vector)

    C = 0
    A = (C*s_vector[-1] - 2*r_end) / s_vector[-1]**3
    B = - (C + 3*A*s_vector[-1]**2) / (2*s_vector[-1])

    r_vector = A*s_vector**3 + B*s_vector**2 + C*s_vector

    grad = np.diff(l1[start_wp_id:start_wp_id+forward_idx], axis=0)
    grad_norm = np.hypot(grad[:, 0], grad[:, 1])
    p_vector = grad[:, [1, 0]] / grad_norm[:, None]  # perpendicular vector

    testl = np.dot(r_temp, p_vector[-1, :] * [-1, 1])
    testr = np.dot(r_temp, p_vector[-1, :] * [1, -1])
    turn_dir = np.argmin([np.arccos(testl), np.arccos(testr)])

    # left
    if turn_dir == 0:
        p_vector[:, 0] *= -1
    # right
    else:
        p_vector[:, 1] *= -1

    lane_change_path = l1[start_wp_id+1:start_wp_id+forward_idx] + p_vector*r_vector[1:, None]
    _, start_id = get_waypoint_pos(lane_change_path[0], l1)
    _, end_id = get_waypoint_pos(lane_change_path[-1], l2)
    new_path = np.concatenate((l1[:start_id], lane_change_path, l2[end_id+1:]))

    return new_path


class Vehicle:
    LENGTH = 5
    WIDTH = 1.8

    def __init__(self, init_state: tuple):
        self.x, self.y, self.h, self.v = init_state

    def veh_shape(self):
        ang = np.arctan2(self.WIDTH / 2, self.LENGTH / 2)
        sd = np.hypot(self.WIDTH / 2, self.LENGTH / 2)
        shape = []
        ang_list = [ang, np.pi - ang, np.pi + ang, -ang]
        for i in range(4):
            temp_ang = ang_list[i]
            shape.append([self.x + sd * np.cos(temp_ang + self.h),
                          self.y + sd * np.sin(temp_ang + self.h)])

        return np.array(shape)

    def pure_pursuit(self, waypoint: numpy.ndarray):
        """
        simple pure pursuit controller for lateral control
        """
        Kg, Kc = 0.3, 4
        rear_pos = np.array([self.x, self.y]) - (self.LENGTH / 4) * np.array([np.cos(self.h), np.sin(self.h)])
        LOOK_AHEAD = Kg * self.v + Kc

        target_x = self.x + LOOK_AHEAD * np.cos(self.h)
        target_y = self.y + LOOK_AHEAD * np.sin(self.h)

        target_pos, _ = get_waypoint_pos([target_x, target_y], waypoint)

        alpha = np.arctan2(target_pos[1] - rear_pos[1], target_pos[0] - rear_pos[0]) - self.h

        l_dist = np.hypot(target_pos[1] - rear_pos[1], target_pos[0] - rear_pos[0])

        delta_steer = np.arctan2(self.LENGTH * np.sin(alpha), l_dist)

        return delta_steer

    def p_controller(self, target_speed: float):
        Kp = 1
        return Kp * (target_speed - self.v)

    def update(self, target_speed: float, waypoint: numpy.ndarray):
        a = self.p_controller(target_speed)
        delta_steer = self.pure_pursuit(waypoint)

        self.x += self.v * np.cos(self.h) * dt
        self.y += self.v * np.sin(self.h) * dt
        self.h += self.v * np.tan(delta_steer) * dt / (self.LENGTH / 2)
        self.v += a * dt


x = np.arange(0, 100, 0.5)

# straight road
# y_1 = np.full(x.shape, 2)
# y_2 = np.full(x.shape, 6)
# lane_1 = np.vstack((x, y_1)).transpose()
# lane_2 = np.vstack((x, y_2)).transpose()

# curved road
p = np.poly1d([0.004, 0, 0])
y_1 = p(x)
lane_1 = np.vstack((x, y_1)).transpose()
lane_1 = equalize_wp_delta(lane_1)
lane_2 = lane_1 + np.array([-1, 4])

dt = 0.1
INIT_STATE = (0, 2, 0, 10)
TARGET_SPEED = 10
STOP = 90

if __name__ == '__main__':
    ego = Vehicle(INIT_STATE)

    START_LANE_CHANGE = 40
    lane_change = False

    while ego.x < STOP:
        if ego.x > START_LANE_CHANGE and not lane_change:
            waypoint = lane_change_waypoint(lane_1, lane_2, [ego.x, ego.y])
            lane_change = True
        if not lane_change:
            waypoint = lane_1

        ego.update(TARGET_SPEED, waypoint)
        plt.gca().cla()
        plt.plot(lane_1[:, 0], lane_1[:, 1], "k--")
        plt.plot(lane_2[:, 0], lane_2[:, 1], "k--")
        plt.plot(waypoint[:, 0], waypoint[:, 1], "r")
        ego_shape = ego.veh_shape()
        plt.fill(ego_shape[:, 0], ego_shape[:, 1], "b")
        plt.ylim((-1, 7))
        plt.axis("equal")
        # display.display(plt.gcf())
        # display.clear_output(wait=True)

        plt.pause(0.01)
