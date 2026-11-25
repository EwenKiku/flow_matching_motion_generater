"""Steer a live flow-matching rollout next to its looping style reference."""
import argparse
from collections import deque
from itertools import pairwise
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import glfw
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(ROOT))

from motion_flow.data import load_clip
from motion_flow.online import TrajectoryGenerator
from motion_flow.robot import DEFAULT_XML, root_quat


DEFAULT_STYLE = ROOT / 'data/pi_plus_motions/walk_hands_on_back_loop_001__A029.npz'
KEY_PAIRS = ((glfw.KEY_W, glfw.KEY_S), (glfw.KEY_A, glfw.KEY_D), (glfw.KEY_Q, glfw.KEY_E))
IDENTITY = np.eye(3).ravel()
ZERO_SIZE = np.zeros(3)


def adjust_command(command, key, steps):
    if key == glfw.KEY_SPACE:
        command.fill(0.)
        return
    for axis, (positive, negative) in enumerate(KEY_PAIRS):
        if key in (positive, negative):
            delta = steps[axis] if key == positive else -steps[axis]
            value = command[axis] + delta
            command[axis] = 0. if abs(value) < steps[axis] * 1e-12 else value
            return


def display_model():
    xml = ET.parse(DEFAULT_XML).getroot()
    compiler = xml.find('compiler')
    compiler.set('meshdir', str((DEFAULT_XML.parent / compiler.get('meshdir')).resolve()))
    asset = xml.find('asset')
    ET.SubElement(asset, 'texture', name='grid', type='2d', builtin='checker', width='128',
                  height='128', rgb1='.28 .34 .37', rgb2='.37 .44 .47')
    ET.SubElement(asset, 'material', name='floor', texture='grid', texrepeat='3 3', texuniform='true')
    ET.SubElement(xml.find('worldbody'), 'geom', type='plane', size='0 0 .01', material='floor')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    model.vis.headlight.ambient[:] = .38
    model.vis.headlight.diffuse[:] = .75
    return model


class SplitView:
    def __init__(self, model, joints, heading):
        self.model = model
        self.data = mujoco.MjData(model)
        self.address = [model.joint(n).qposadr.item() for n in joints]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0  # Hide collision primitives.
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = np.degrees(heading) + 130
        self.camera.elevation = -14
        self.camera.distance = 1.45
        self.pan_offset = np.zeros(3)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def bind_mouse(self, window):
        """Apply the same orbit, pan and zoom to both robot-following views."""
        drag_button = None
        last_x = last_y = 0.

        def on_button(window, button, action, mods):
            nonlocal drag_button, last_x, last_y
            if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
                return
            if action == glfw.PRESS:
                drag_button = button
                last_x, last_y = glfw.get_cursor_pos(window)
            elif button == drag_button:
                drag_button = None

        def on_motion(window, x, y):
            nonlocal last_x, last_y
            if drag_button is None:
                return
            height = max(1, glfw.get_window_size(window)[1])
            dx, dy = (x - last_x) / height, (y - last_y) / height
            last_x, last_y = x, y
            camera = self.camera
            if drag_button == glfw.MOUSE_BUTTON_LEFT:
                camera.azimuth = (camera.azimuth - 180 * dx) % 360
                camera.elevation = np.clip(camera.elevation - 180 * dy, -89.9, 89.9)
            else:
                az, el = np.radians([camera.azimuth, camera.elevation])
                right = np.array([np.sin(az), -np.cos(az), 0.])
                up = np.array([-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)])
                span = 2 * camera.distance * np.tan(np.radians(self.model.vis.global_.fovy) / 2)
                self.pan_offset += span * (dy * up - dx * right)

        def on_scroll(window, xoffset, yoffset):
            self.camera.distance = max(.02, self.camera.distance * .90 ** yoffset)

        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_cursor_pos_callback(window, on_motion)
        glfw.set_scroll_callback(window, on_scroll)

    def draw(self, viewport, frame, trail, color, title, status):
        self.data.qpos[:7] = frame[:7]
        self.data.qpos[self.address] = frame[7:]
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.pan_offset + (frame[0], frame[1], .31)
        mujoco.mjv_updateScene(self.model, self.data, self.options, None, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        for start, end in pairwise(trail):
            if np.array_equal(start, end):
                continue
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, ZERO_SIZE, start, IDENTITY, color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3., start, end)
            self.scene.ngeom += 1
        mujoco.mjr_render(viewport, self.scene, self.context)
        font = mujoco.mjtFont.mjFONT_NORMAL
        mujoco.mjr_overlay(font, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, title, '', self.context)
        mujoco.mjr_overlay(font, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, status, '', self.context)


def run_viewer(window, view, generator, clip, style_name, speed_step):
    reference = np.column_stack((clip.xy, clip.pose[:, 0],
                                 root_quat(clip.pose[:, 1], clip.pose[:, 2], clip.yaw), clip.pose[:, 3:]))
    index = min(generator.cfg.history, len(reference)) - 1
    generated = reference[index]
    trails = (deque(maxlen=40), deque(maxlen=40))
    colors = (np.array([.26, .82, 1., 1.], dtype='f'), np.array([1., .66, .15, 1.], dtype='f'))
    ticks = 0
    period = 1. / generator.fps
    cmd = np.zeros(3)

    def on_key(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            else:
                adjust_command(cmd, key, speed_step)

    def on_focus(window, focused):
        if not focused:
            cmd.fill(0.)

    glfw.set_key_callback(window, on_key)
    glfw.set_window_focus_callback(window, on_focus)
    view.bind_mouse(window)
    deadline = time.monotonic()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.window_should_close(window):
            break
        now = time.monotonic()
        if now < deadline:
            glfw.wait_events_timeout(deadline - now)
            continue

        # Advance on the motion clock. Bound catch-up after long window-system stalls.
        deadline = max(deadline, now - 3 * period)
        while deadline <= now:
            generated = generator.step(cmd)
            index = (index + 1) % len(reference)
            if index == 0:
                trails[0].clear()
            if ticks % 4 == 0:
                for trail, frame in zip(trails, (reference[index], generated)):
                    trail.append(np.array([frame[0], frame[1], .004]))
            ticks += 1
            deadline += period

        width, height = glfw.get_framebuffer_size(window)
        if width >= 2 and height > 0:
            mid = width // 2
            left = mujoco.MjrRect(0, 0, mid, height)
            right = mujoco.MjrRect(mid, 0, width-mid, height)
            view.draw(left, reference[index], trails[0], colors[0], 'Style reference',
                      f'{style_name}\nFrame {index+1} / {len(reference)}')
            view.draw(right, generated, trails[1], colors[1], 'Live flow matching',
                      f'vx {cmd[0]:+.2f} m/s    vy {cmd[1]:+.2f} m/s    wz {cmd[2]:+.2f} rad/s\n'
                      'Tap W/S: vx +/-    A/D: vy +/-    Q/E: wz +/-\nSpace: zero speeds    Esc: quit')
            glfw.swap_buffers(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', type=Path, default=DEFAULT_STYLE)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/pi_plus_walk.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--speed-step', nargs=3, type=float, default=(.05, .05, .10),
                        metavar=('FORWARD', 'LATERAL', 'TURN'), help='Speed change per tap in m/s, m/s, rad/s')
    args = parser.parse_args()
    if not np.isfinite(args.speed_step).all() or min(args.speed_step) <= 0:
        parser.error('--speed-step requires three finite, positive values')
    torch.set_num_threads(1)
    generator = TrajectoryGenerator(args.checkpoint, args.style, device=args.device, seed=42)
    clip = load_clip(args.style, generator.robot.joints, generator.fps)
    model = display_model()
    if not glfw.init():
        raise SystemExit('Cannot initialize GLFW. Run the viewer in a graphical desktop session.')
    try:
        window = glfw.create_window(1200, 640, 'Pi Plus | Style reference and live flow matching', None, None)
        if not window:
            raise SystemExit('Cannot create the MuJoCo OpenGL window.')
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        view = SplitView(model, generator.robot.joints, clip.yaw[min(generator.cfg.history, len(clip.yaw))-1])
        try:
            run_viewer(window, view, generator, clip, args.style.stem, args.speed_step)
        finally:
            view.context.free()
    finally:
        glfw.terminate()


if __name__ == '__main__':
    main()
