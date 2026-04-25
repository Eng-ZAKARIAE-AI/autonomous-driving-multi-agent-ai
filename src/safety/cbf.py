import numpy as np

class ControlBarrierFunction:
    def __init__(self, safety_margin=2.5, alpha=0.5):
        self.d_safe = safety_margin [cite: 323]
        self.alpha = alpha [cite: 323]

    def h(self, ego, obstacle):
        # Distance euclidienne - marge de sécurité
        dist = np.sqrt((ego.x - obstacle.x)**2 + (ego.y - obstacle.y)**2) - self.d_safe [cite: 323]
        return dist
