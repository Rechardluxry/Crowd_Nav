import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class MyPolicy(Policy):
    """
    Skeleton policy for humans.
    Replace predict() with your own logic.
    """
    def __init__(self):
        super().__init__()
        self.name = 'MY_POLICY'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.time_step = None

    def configure(self, config):
        # If you need custom params, read them from config here.
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        state.self_state: FullState of the human
        state.human_states: list of ObservableState for other agents (and possibly robot)
        """
        self_state = state.self_state
        # Simple goal-seeking baseline: move straight toward goal at preferred speed.
        goal_vec = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(goal_vec)
        if speed > 1e-6:
            pref_vel = goal_vec / speed
        else:
            pref_vel = np.zeros(2)
        action = ActionXY(pref_vel[0] * self_state.v_pref, pref_vel[1] * self_state.v_pref)
        self.last_state = state
        return action
