import numpy as np
import pinocchio as pin


class MPCPredictionRecorder:
    def __init__(self, model, foot_frame_names):
        self.model = model
        self.data = pin.Data(model)
        self.foot_frame_names = list(foot_frame_names)
        self.foot_frame_ids = []
        for name in self.foot_frame_names:
            frame_id = model.getFrameId(name)
            if frame_id >= len(model.frames):
                raise ValueError(f"Unknown frame name: {name}")
            self.foot_frame_ids.append(frame_id)
        self.joint_names = list(model.names)[2:]
        self._times = []
        self._com_horizon = []
        self._q_horizon = []
        self._feet_horizon = []

    def append(self, t, xs):
        xs = np.asarray(xs)
        if xs.ndim != 2:
            raise ValueError("xs must be 2D array-like (H, nx)")
        nq = self.model.nq
        q_h = xs[:, :nq]
        com_h = np.zeros((q_h.shape[0], 3))
        feet_h = np.zeros((q_h.shape[0], len(self.foot_frame_ids), 3))
        for k in range(q_h.shape[0]):
            q = q_h[k]
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            com_h[k] = pin.centerOfMass(self.model, self.data, q)
            feet_h[k] = np.stack(
                [self.data.oMf[fid].translation.copy() for fid in self.foot_frame_ids],
                axis=0,
            )
        self._times.append(float(t))
        self._com_horizon.append(com_h)
        self._q_horizon.append(q_h.copy())
        self._feet_horizon.append(feet_h)

    def __len__(self):
        return len(self._times)

    def save(self, path, horizon_dt):
        times = np.asarray(self._times, dtype=float)
        com = np.asarray(self._com_horizon, dtype=float)
        q = np.asarray(self._q_horizon, dtype=float)
        feet = np.asarray(self._feet_horizon, dtype=float)
        joint_q = q[:, :, 7:] if q.ndim == 3 and q.shape[2] >= 7 else q.copy()
        np.savez_compressed(
            path,
            version=np.array([2], dtype=np.int32),
            times=times,
            horizon_dt=np.array([float(horizon_dt)], dtype=float),
            com_horizon=com,
            q_horizon=q,
            joint_q_horizon=joint_q,
            foot_pos_horizon=feet,
            foot_frame_names=np.array(self.foot_frame_names, dtype=object),
            joint_names=np.array(self.joint_names, dtype=object),
        )
