import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _prompt_choice(prompt, default):
    value = input(prompt).strip()
    return value if value else default


def _parse_select_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def _select_joint_indices(text, joint_names):
    if not text:
        return list(range(len(joint_names)))
    indices = []
    for item in _parse_select_list(text):
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(joint_names):
                indices.append(idx)
            else:
                print(f"[warn] joint index out of range: {idx}")
        else:
            if item in joint_names:
                indices.append(joint_names.index(item))
            else:
                print(f"[warn] unknown joint name: {item}")
    return indices


class FramePlayer:
    def __init__(
        self,
        times,
        horizon_dt,
        com_horizon,
        foot_pos_horizon,
        joint_q_horizon,
        joint_names,
        show_com,
        show_feet,
        show_joints,
        joint_indices,
    ):
        self.times = times
        self.horizon_dt = horizon_dt
        self.com_horizon = com_horizon
        self.foot_pos_horizon = foot_pos_horizon
        self.joint_q_horizon = joint_q_horizon
        self.joint_names = joint_names
        self.show_com = show_com
        self.show_feet = show_feet
        self.show_joints = show_joints
        self.joint_indices = joint_indices
        self.n = len(times)
        self.h = com_horizon.shape[1] if com_horizon is not None else foot_pos_horizon.shape[1]
        self.i = 0

        ncols = 2 if show_joints else 1
        self.fig = plt.figure(figsize=(10, 5))
        self._init_axes(ncols)
        self._init_artists()
        self._connect()
        self.update()

    def _init_axes(self, ncols):
        self.ax3d = None
        self.axj = None
        if self.show_com or self.show_feet:
            self.ax3d = self.fig.add_subplot(1, ncols, 1, projection="3d")
        if self.show_joints:
            self.axj = self.fig.add_subplot(1, ncols, ncols)

        if self.ax3d is not None:
            pos = []
            if self.show_com:
                pos.append(self.com_horizon.reshape(-1, 3))
            if self.show_feet:
                pos.append(self.foot_pos_horizon.reshape(-1, 3))
            if pos:
                all_pos = np.vstack(pos)
                finite = np.isfinite(all_pos).all(axis=1)
                if np.any(finite):
                    all_pos = all_pos[finite]
                mins = np.min(all_pos, axis=0) if all_pos.size else np.array([-1.0, -1.0, -1.0])
                maxs = np.max(all_pos, axis=0) if all_pos.size else np.array([1.0, 1.0, 1.0])
                pad = np.maximum(0.2, 0.1 * (maxs - mins + 1e-6))
                self.ax3d.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
                self.ax3d.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
                self.ax3d.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
            self.ax3d.set_xlabel("x")
            self.ax3d.set_ylabel("y")
            self.ax3d.set_zlabel("z")
            self.ax3d.set_title("CoM / Feet (horizon)")

        if self.axj is not None:
            self.axj.set_title("Joint positions (horizon)")
            self.axj.set_xlabel("t in horizon")
            self.axj.set_ylabel("q")

        slider_ax = self.fig.add_axes([0.15, 0.02, 0.7, 0.04])
        self.slider = Slider(
            ax=slider_ax,
            label="frame",
            valmin=0,
            valmax=max(self.n - 1, 0),
            valinit=0,
            valstep=1,
        )

    def _init_artists(self):
        self.com_line = None
        self.com_lc = None
        self.feet_lines = []
        self.feet_lcs = []
        if self.ax3d is not None:
            if self.show_com:
                dummy = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
                self.com_lc = Line3DCollection(dummy, cmap="viridis")
                self.ax3d.add_collection3d(self.com_lc)
            if self.show_feet:
                nfeet = self.foot_pos_horizon.shape[2]
                for i in range(nfeet):
                    dummy = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
                    lc = Line3DCollection(dummy, cmap="plasma")
                    self.ax3d.add_collection3d(lc)
                    self.feet_lcs.append(lc)
            self.ax3d.legend(loc="upper right")

        self.joint_lines = []
        self.joint_lcs = []
        if self.axj is not None:
            dt = self.horizon_dt if self.horizon_dt > 0.0 else 1.0
            t = np.arange(self.h) * dt
            for idx in self.joint_indices:
                lc = LineCollection([], cmap="cividis")
                self.axj.add_collection(lc)
                self.joint_lcs.append(lc)
            self.axj.legend(loc="upper right")

    def _connect(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.slider.on_changed(self.on_slider)

    def on_key(self, event):
        if event.key in ("right", "d"):
            self.step(1)
        elif event.key in ("left", "a"):
            self.step(-1)
        elif event.key in ("up",):
            self.step(10)
        elif event.key in ("down",):
            self.step(-10)
        elif event.key in ("home",):
            self.set_index(0)
        elif event.key in ("end",):
            self.set_index(self.n - 1)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def on_slider(self, val):
        self.set_index(int(val))

    def step(self, delta):
        self.set_index(self.i + delta)

    def set_index(self, idx):
        self.i = int(np.clip(idx, 0, self.n - 1))
        self.update()

    def update(self):
        t = self.times[self.i]
        if self.com_line is not None:
            com = self.com_horizon[self.i]
            if np.isfinite(com).all():
                segs = np.stack([com[:-1], com[1:]], axis=1)
                self.com_lc.set_segments(segs)
                self.com_lc.set_array(np.linspace(0.0, 1.0, segs.shape[0]))
        if self.feet_lcs:
            feet = self.foot_pos_horizon[self.i]
            for idx, lc in enumerate(self.feet_lcs):
                if idx >= feet.shape[1]:
                    break
                traj = feet[:, idx, :]
                if np.isfinite(traj).all():
                    segs = np.stack([traj[:-1], traj[1:]], axis=1)
                    lc.set_segments(segs)
                    lc.set_array(np.linspace(0.0, 1.0, segs.shape[0]))
        if self.joint_lcs:
            q = self.joint_q_horizon[self.i]
            dt = self.horizon_dt if self.horizon_dt > 0.0 else 1.0
            t = np.arange(self.h) * dt
            for lc, idx in zip(self.joint_lcs, self.joint_indices):
                y = q[:, idx]
                if np.isfinite(y).all():
                    pts = np.column_stack([t, y])
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    lc.set_segments(segs)
                    lc.set_array(np.linspace(0.0, 1.0, segs.shape[0]))
            self.axj.relim()
            self.axj.autoscale_view(scalex=False, scaley=True)
        self.fig.suptitle(f"cycle {self.i + 1}/{self.n}  t={t:.4f}s")
        if int(self.slider.val) != self.i:
            self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", help="path to .npz recording")
    parser.add_argument("--show", default="", help="comma list: com,feet,joints")
    parser.add_argument("--joints", default="", help="comma list of joint indices or names")
    args = parser.parse_args()

    data = np.load(args.recording, allow_pickle=True)
    times = data["times"]
    if "com_horizon" in data:
        com_horizon = data["com_horizon"]
        foot_pos_horizon = data["foot_pos_horizon"]
        joint_q_horizon = data["joint_q_horizon"]
        horizon_dt = float(np.asarray(data["horizon_dt"])[0])
    else:
        print("[warn] recording is single-step (version 1). Re-run simulation to record full horizon.")
        com = data["com"]
        foot_pos = data["foot_pos"]
        joint_q = data["joint_q"]
        com_horizon = com[:, None, :]
        foot_pos_horizon = foot_pos[:, None, :, :]
        joint_q_horizon = joint_q[:, None, :]
        horizon_dt = 0.0
    joint_names = [str(n) for n in data["joint_names"].tolist()]
    print(f"[recording] cycles={len(times)} horizon={com_horizon.shape[1]}")
    print(f"[recording] com_horizon shape={com_horizon.shape} feet shape={foot_pos_horizon.shape}")
    finite_ratio = np.isfinite(com_horizon).mean()
    print(f"[recording] com finite ratio={finite_ratio:.3f}")
    if len(times) > 0:
        np.set_printoptions(precision=6, suppress=True)
        print("[recording] first cycle time", float(times[0]))
        print("[recording] first cycle com_horizon")
        print(com_horizon[0])
        print("[recording] first cycle foot_pos_horizon")
        print(foot_pos_horizon[0])
        print("[recording] first cycle joint_q_horizon")
        print(joint_q_horizon[0])

    show = _parse_select_list(args.show) if args.show else None
    if show is None:
        print("Available: com, feet, joints")
        show = _parse_select_list(_prompt_choice("Select content (default: com,feet): ", "com,feet"))
    show = {s.lower() for s in show}
    show_com = "com" in show
    show_feet = "feet" in show
    show_joints = "joints" in show

    joint_indices = []
    if show_joints:
        print(f"Total joints: {len(joint_names)}")
        print("Examples: 0,1,2 or Hip1Joint (name depends on model)")
        joint_text = args.joints if args.joints else _prompt_choice("Select joints (default: all): ", "")
        joint_indices = _select_joint_indices(joint_text, joint_names)
        if not joint_indices:
            joint_indices = list(range(len(joint_names)))

    if not (show_com or show_feet or show_joints):
        print("Nothing selected to show. Use --show com,feet,joints")
        return

    print("Controls: left/right=step, up/down=jump 10, home/end, q=quit")
    FramePlayer(
        times=times,
        horizon_dt=horizon_dt,
        com_horizon=com_horizon,
        foot_pos_horizon=foot_pos_horizon,
        joint_q_horizon=joint_q_horizon,
        joint_names=joint_names,
        show_com=show_com,
        show_feet=show_feet,
        show_joints=show_joints,
        joint_indices=joint_indices,
    )
    plt.show()


if __name__ == "__main__":
    main()
