    def pure_pursuit(state):
        Kg, Kc = 0.3, 4
        rear_pos = np.array(ego_state[:2]) - 0.8 * ego_state[4] * np.array([np.cos(ego_state[3]),
                                                                                      np.sin(ego_state[3])])
        LOOK_AHEAD = Kg * ego_state[2] + Kc
        f_state = to_frenet2(np.array([[rear_pos[0], rear_pos[1]]]), self.reference_path[:, :2])[0]
        s, r, idx = f_state[0], f_state[1], f_state[2].astype('int')
        x_ref, y_ref, _ = to_cartesian(np.array([s + LOOK_AHEAD]), np.array([0]), self.reference_path[:, :2], idx)
        h = np.arctan2(y_ref-rear_pos[1], x_ref-rear_pos[0]) #- self.ego_state[3]
        # print(h, self.ego_state[:4], rear_pos, [x_ref, y_ref])
        return h
