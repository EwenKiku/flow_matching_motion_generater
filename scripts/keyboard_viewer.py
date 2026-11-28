"""Steer a live flow-matching rollout next to its looping style reference."""
import argparse
import base64
import os
from collections import deque
from itertools import pairwise
from pathlib import Path
import struct
import sys
import time
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def forwarded_x11():
    """ssh -X sets DISPLAY to localhost:10.0. A local desktop stays on :0 and keeps GLFW."""
    host = os.environ.get('DISPLAY', '').split(':', 1)[0]
    return host in ('localhost', '127.0.0.1')


# SSH X11 forwarding can show a window, but GLFW's GLX context crashes there.
FORWARDED_X11 = forwarded_x11()
if FORWARDED_X11:
    os.environ['MUJOCO_GL'] = 'egl'

import glfw
import mujoco
import numpy as np
import torch

from motion_flow.data import load_clip, measured_twist
from motion_flow.robot import root_quat
from scripts.config import add_robot_arguments, make_generator, resolve_robot


KEY_PAIRS = ((glfw.KEY_W, glfw.KEY_S), (glfw.KEY_A, glfw.KEY_D), (glfw.KEY_Q, glfw.KEY_E))
IDENTITY = np.eye(3).ravel()
ZERO_SIZE = np.zeros(3)


class VelocityPlots:
    def __init__(self, fps):
        self.seconds = 10.
        self.figures = []
        for title in ('vx m/s', 'vy m/s', 'wz rad/s'):
            fig = mujoco.MjvFigure()
            fig.title = title
            fig.xlabel = 't (s)'
            fig.xformat, fig.yformat = '%.0f', '%.2f'
            fig.minwidth = '-0.00'
            fig.gridsize[:] = (4, 4)
            fig.flg_extend = 0
            fig.flg_legend = 0
            fig.flg_ticklabel[:] = 1
            fig.linergb[:2] = [[1., .68, .24], [.25, .85, .95]]
            fig.linewidth = 2.
            self.figures.append(fig)
        capacity = min(round(self.seconds * fps) + 1, self.figures[0].linedata.shape[1] // 2)
        self.samples = deque(maxlen=capacity)
        self.marker_stride = max(1, round(.5 * fps))

    def append(self, timestamp, command, measured):
        self.samples.append((timestamp, *command, *measured))

    def draw(self, width, height, context):
        if len(self.samples) < 2:
            return
        samples = np.asarray(self.samples)
        times = samples[:, 0]
        command, measured = samples[-1, 1:4], samples[-1, 4:7]
        end = max(self.seconds, times[-1])
        count = len(samples)
        legend_height = min(32, height // 5)
        for i, label in enumerate(('cmd', 'root')):
            left = width * i // 2
            rect = mujoco.MjrRect(left, height - legend_height, width * (i+1) // 2 - left, legend_height)
            mujoco.mjr_label(rect, mujoco.mjtFont.mjFONT_NORMAL, label, .07, .10, .12, 1.,
                             *self.figures[0].linergb[i], context)
        for axis, fig in enumerate(self.figures):
            values = samples[:, [axis + 1, axis + 4]]
            low, high = min(0., values.min()), max(0., values.max())
            pad = max(.05, .25 * (high - low))
            fig.title = (f'{("vx m/s", "vy m/s", "wz rad/s")[axis]}'
                         f'  cmd {command[axis]:+.2f}  root {measured[axis]:+.2f}')
            fig.range[:] = ((end - self.seconds, end), (low - pad, high + pad))
            fig.linepnt[:2] = count
            fig.linedata[:2, :2*count:2] = times
            fig.linedata[:2, 1:2*count:2] = values.T
            left = axis * width // 3
            rect = mujoco.MjrRect(left, 0, (axis+1)*width//3 - left, height - legend_height)
            mujoco.mjr_figure(rect, fig, context)
            # Mark measured samples so both series remain visible at exact agreement.
            xp, yp = fig.xaxispixel, fig.yaxispixel
            if xp[1] <= xp[0] or yp[1] <= yp[0]:
                continue
            for i in range(count-1, -1, -self.marker_stride):
                x = np.interp(times[i], fig.xaxisdata, xp)
                y = np.interp(values[i, 1], fig.yaxisdata, yp)
                mujoco.mjr_rectangle(mujoco.MjrRect(round(x)-2, round(y)-2, 5, 5),
                                     *fig.linergb[1], 1.)


class StyleReel:
    """NPZ files beside the current reference. Left and right arrow keys move through them."""

    def __init__(self, path):
        path = Path(path).resolve()
        self.files = sorted(path.parent.glob('*.npz'))
        if path not in self.files:
            self.files.append(path)
            self.files.sort()
        self.index = self.files.index(path)

    @property
    def path(self):
        return self.files[self.index]

    @property
    def name(self):
        return self.path.stem

    def move(self, step):
        self.index = (self.index + step) % len(self.files)
        return self.path


def reference_frames(clip):
    return np.column_stack((clip.xy, clip.pose[:, 0],
                            root_quat(clip.pose[:, 1], clip.pose[:, 2], clip.yaw), clip.pose[:, 3:]))


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


def display_model(xml_path):
    xml = ET.parse(xml_path).getroot()
    compiler = xml.find('compiler')
    compiler.set('meshdir', str((xml_path.parent / compiler.get('meshdir')).resolve()))
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
    def __init__(self, model, joints, heading, viewer):
        self.model = model
        self.lookat_height = viewer.lookat_height
        self.data = mujoco.MjData(model)
        self.address = [model.joint(n).qposadr.item() for n in joints]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0  # Hide collision primitives.
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = np.degrees(heading) + viewer.azimuth_offset
        self.camera.elevation = viewer.elevation
        self.camera.distance = viewer.distance
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
                last_x, last_y = glfw.get_cursor_pos(window)
                drag_button = button if last_y < glfw.get_window_size(window)[1] * 2/3 else None
            elif button == drag_button:
                drag_button = None

        def on_motion(window, x, y):
            nonlocal last_x, last_y
            if drag_button is None:
                return
            height = max(1, glfw.get_window_size(window)[1] * 2/3)
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
            if glfw.get_cursor_pos(window)[1] < glfw.get_window_size(window)[1] * 2/3:
                self.camera.distance = max(.02, self.camera.distance * .90 ** yoffset)

        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_cursor_pos_callback(window, on_motion)
        glfw.set_scroll_callback(window, on_scroll)

    def draw(self, viewport, frame, trail, color, title, status):
        self.data.qpos[:7] = frame[:7]
        self.data.qpos[self.address] = frame[7:]
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.pan_offset + (frame[0], frame[1], self.lookat_height)
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


def run_viewer(window, view, generator, reel, speed_step):
    reference = reference_frames(load_clip(
        reel.path, generator.robot.joints, generator.fps, generator.anchor_body))
    index = min(generator.cfg.history, len(reference)) - 1
    generated = reference[index]
    trails = (deque(maxlen=40), deque(maxlen=40))
    colors = (np.array([.26, .82, 1., 1.], dtype='f'), np.array([1., .66, .15, 1.], dtype='f'))
    ticks = 0
    period = 1. / generator.fps
    plots = VelocityPlots(generator.fps)
    cmd = np.zeros(3)
    measured = np.zeros(3)

    def switch_style(step):
        nonlocal reference, index
        reference = reference_frames(generator.set_style(reel.move(step)))
        index = 0
        trails[0].clear()

    def on_key(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_LEFT:
            switch_style(-1)
        elif key == glfw.KEY_RIGHT:
            switch_style(1)
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
            previous = generated
            generated = generator.step(cmd)
            measured = measured_twist(previous, generated, period)
            plots.append((ticks + 1) * period, cmd, measured)
            index = (index + 1) % len(reference)
            if index == 0:
                trails[0].clear()
            if ticks % 4 == 0:
                for trail, frame in zip(trails, (reference[index], generated)):
                    trail.append(np.array([frame[0], frame[1], .004]))
            ticks += 1
            deadline += period

        width, height = glfw.get_framebuffer_size(window)
        if width >= 3 and height >= 3:
            mid = width // 2
            plot_height = height // 3
            left = mujoco.MjrRect(0, plot_height, mid, height - plot_height)
            right = mujoco.MjrRect(mid, plot_height, width-mid, height - plot_height)
            view.draw(left, reference[index], trails[0], colors[0], reel.name,
                      f'风格参考\n帧 {index + 1} / {len(reference)}\n左右方向键切换')
            view.draw(right, generated, trails[1], colors[1], reel.name,
                      '实时生成\n'
                      f'指令  vx {cmd[0]:+.3f}   vy {cmd[1]:+.3f}   wz {cmd[2]:+.3f}\n'
                      f'根部  vx {measured[0]:+.3f}   vy {measured[1]:+.3f}   wz {measured[2]:+.3f}')
            plots.draw(width, plot_height, view.context)
            glfw.swap_buffers(window)


def rgb_png(rgb):
    """Uncompressed-filter PNG so Tk can show an EGL frame without a GLX window."""
    height, width = rgb.shape[:2]
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(height))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 1))
            + chunk(b'IEND', b''))


class ForwardedView:
    """GPU render via EGL, then hand the picture to a plain X11 window."""

    def __init__(self, model, joints, heading, viewer):
        self.model = model
        self.lookat_height = viewer.lookat_height
        self.data = mujoco.MjData(model)
        self.address = [model.joint(n).qposadr.item() for n in joints]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = np.degrees(heading) + viewer.azimuth_offset
        self.camera.elevation = viewer.elevation
        self.camera.distance = viewer.distance
        self.pan_offset = np.zeros(3)
        self.renderer = mujoco.Renderer(model, height=360, width=480)

    def render(self, frame, trail, color):
        self.data.qpos[:7] = frame[:7]
        self.data.qpos[self.address] = frame[7:]
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.pan_offset + (frame[0], frame[1], self.lookat_height)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        scene = self.renderer.scene
        for start, end in pairwise(trail):
            if np.array_equal(start, end) or scene.ngeom >= scene.maxgeom:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, ZERO_SIZE, start, IDENTITY, color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3., start, end)
            scene.ngeom += 1
        return self.renderer.render()

    def orbit(self, dx, dy, height):
        height = max(1., height)
        self.camera.azimuth = (self.camera.azimuth - 180 * dx / height) % 360
        self.camera.elevation = np.clip(self.camera.elevation - 180 * dy / height, -89.9, 89.9)

    def pan(self, dx, dy, height):
        height = max(1., height)
        dx, dy = dx / height, dy / height
        az, el = np.radians([self.camera.azimuth, self.camera.elevation])
        right = np.array([np.sin(az), -np.cos(az), 0.])
        up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
        span = 2 * self.camera.distance * np.tan(np.radians(self.model.vis.global_.fovy) / 2)
        self.pan_offset += span * (dy * up - dx * right)

    def zoom(self, inward):
        factor = .90 if inward else 1 / .90
        self.camera.distance = max(.02, self.camera.distance * factor)


def axis_ticks(low, high, count=4):
    """A few readable y ticks that stay inside the plot range and include zero."""
    span = high - low
    raw = span / count
    power = 10 ** np.floor(np.log10(raw))
    step = next(factor * power for factor in (1, 2, 2.5, 5, 10) if span / (factor * power) <= count + 1)
    first = np.ceil((low + span * 1e-6) / step) * step
    ticks = []
    value = first
    while value < high - span * 1e-6:
        ticks.append(0. if abs(value) <= step * 1e-6 else round(float(value), 6))
        value += step
    if low < 0 < high and all(tick != 0. for tick in ticks):
        ticks.append(0.)
    return ticks


def speed_text(prefix, values):
    return (f'{prefix}    vx {values[0]:+.3f} m/s      vy {values[1]:+.3f} m/s'
            f'      wz {values[2]:+.3f} rad/s')


class TkVelocityPlots:
    """Command and measured root twist for the last 10 seconds, drawn under the robots."""

    def __init__(self, fps):
        self.seconds = 10.
        self.samples = deque(maxlen=round(self.seconds * fps) + 1)
        self.titles = ('前向 vx  m/s', '侧向 vy  m/s', '偏航 wz  rad/s')
        self.command = '#ffad3d'
        self.measured = '#40d9f2'

    def append(self, timestamp, command, measured):
        self.samples.append((timestamp, *command, *measured))

    def draw(self, canvas):
        canvas.delete('all')
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 100:
            width = 960
        if height < 40:
            height = 230
        canvas.configure(bg='#12181e')
        if len(self.samples) < 2:
            return
        samples = np.asarray(self.samples)
        times = samples[:, 0]
        command, measured = samples[-1, 1:4], samples[-1, 4:7]
        end = max(self.seconds, times[-1])
        start = end - self.seconds
        font = ('Noto Sans CJK SC', 11)
        top, bottom = 50, height - 8
        for axis, title in enumerate(self.titles):
            left = axis * width // 3 + 64
            right = (axis + 1) * width // 3 - 14
            values = samples[:, [axis + 1, axis + 4]]
            low, high = min(0., values.min()), max(0., values.max())
            pad = max(.05, .25 * (high - low))
            low, high = low - pad, high + pad
            canvas.create_text(left, 12, anchor='w', fill='#d5dde4', text=title, font=font)
            canvas.create_text(left, 32, anchor='w', fill=self.command,
                               text=f'指令 {command[axis]:+.3f}', font=font)
            canvas.create_text(left + 128, 32, anchor='w', fill=self.measured,
                               text=f'机器人 {measured[axis]:+.3f}', font=font)
            canvas.create_rectangle(left, top, right, bottom, outline='#2a3640')
            for tick in axis_ticks(low, high):
                y = top + (high - tick) / (high - low) * (bottom - top)
                canvas.create_line(left, y, right, y, fill='#24303a' if tick else '#3d4c58')
                canvas.create_text(left - 8, y, anchor='e', fill='#c5d0d8', text=f'{tick:+.2f}', font=font)

            def xy(series, left=left, right=right, low=low, high=high):
                px = left + (times - start) / self.seconds * (right - left)
                py = top + (high - series) / (high - low) * (bottom - top)
                return np.column_stack((px, py)).ravel().tolist()

            canvas.create_line(*xy(values[:, 0]), fill=self.command, width=2)
            canvas.create_line(*xy(values[:, 1]), fill=self.measured, width=2)


def run_forwarded_viewer(title, view, generator, reel, speed_step):
    import tkinter as tk
    print('SSH X11 转发不能创建 GLFW 的 OpenGL 窗口。画面在本机 GPU 上渲染，再送到当前屏幕。', flush=True)
    reference = reference_frames(load_clip(
        reel.path, generator.robot.joints, generator.fps, generator.anchor_body))
    index = min(generator.cfg.history, len(reference)) - 1
    generated = reference[index]
    trails = (deque(maxlen=40), deque(maxlen=40))
    colors = (np.array([.26, .82, 1., 1.], dtype='f'), np.array([1., .66, .15, 1.], dtype='f'))
    ticks = 0
    period = 1. / generator.fps
    cmd = np.zeros(3)
    measured = np.zeros(3)
    plots = TkVelocityPlots(generator.fps)
    bg = '#12181e'
    root = tk.Tk()
    root.title(title)
    root.configure(bg=bg)
    font = 'Noto Sans CJK SC'
    hint = tk.Label(root, bg=bg, fg='#8b98a5', font=(font, 11),
                    text='← → 切换风格     W/S 前后     A/D 左右     Q/E 转向     空格归零     Esc 退出')
    hint.pack(fill='x', padx=16, pady=(10, 0))
    panels = tk.Frame(root, bg=bg)
    panels.pack(fill='x', padx=12, pady=8)
    images = []
    names = []
    photos = [None, None]
    for role, accent in (('风格参考', '#40d9f2'), ('实时生成', '#ffad3d')):
        column = tk.Frame(panels, bg=bg)
        column.pack(side='left', expand=True, fill='both', padx=6)
        tk.Label(column, text=role, bg=bg, fg=accent, font=(font, 12)).pack(anchor='w')
        name = tk.Label(column, text=reel.name, bg=bg, fg='#f4f7fa', font=(font, 16))
        name.pack(anchor='w', pady=(0, 6))
        image = tk.Label(column, bg='#0c1014', bd=0)
        image.pack()
        names.append(name)
        images.append(image)
    readout = tk.Frame(root, bg=bg)
    readout.pack(fill='x', padx=18, pady=(4, 0))
    command_text = tk.Label(readout, bg=bg, fg='#ffad3d', anchor='w', font=(font, 15),
                            text=speed_text('指令', cmd))
    command_text.pack(fill='x')
    measured_text = tk.Label(readout, bg=bg, fg='#40d9f2', anchor='w', font=(font, 15),
                             text=speed_text('机器人', measured))
    measured_text.pack(fill='x')
    chart = tk.Canvas(root, bg=bg, height=230, highlightthickness=0)
    chart.pack(fill='x', padx=12, pady=(4, 12))
    keys = {name: code for name, code in (
        ('w', glfw.KEY_W), ('s', glfw.KEY_S), ('a', glfw.KEY_A), ('d', glfw.KEY_D),
        ('q', glfw.KEY_Q), ('e', glfw.KEY_E), ('space', glfw.KEY_SPACE))}

    def switch_style(step):
        nonlocal reference, index
        reference = reference_frames(generator.set_style(reel.move(step)))
        index = 0
        trails[0].clear()
        for name in names:
            name.configure(text=reel.name)

    def on_key(event):
        if event.keysym == 'Escape':
            root.destroy()
        elif event.keysym == 'Left':
            switch_style(-1)
        elif event.keysym == 'Right':
            switch_style(1)
        else:
            code = keys.get(event.keysym.lower())
            if code is not None:
                adjust_command(cmd, code, speed_step)

    def on_press(event):
        event.widget._drag = (event.num, event.x, event.y)

    def on_drag(event):
        drag = getattr(event.widget, '_drag', None)
        if drag is None:
            return
        button, last_x, last_y = drag
        event.widget._drag = (button, event.x, event.y)
        if button == 1:
            view.orbit(event.x - last_x, event.y - last_y, event.widget.winfo_height())
        elif button == 3:
            view.pan(event.x - last_x, event.y - last_y, event.widget.winfo_height())

    def on_wheel(event):
        view.zoom(event.num == 4 or event.delta > 0)

    root.bind('<KeyPress>', on_key)
    root.bind('<FocusOut>', lambda _event: cmd.fill(0.))
    for image in images:
        image.bind('<ButtonPress-1>', on_press)
        image.bind('<ButtonPress-3>', on_press)
        image.bind('<B1-Motion>', on_drag)
        image.bind('<B3-Motion>', on_drag)
        image.bind('<Button-4>', on_wheel)
        image.bind('<Button-5>', on_wheel)
        image.bind('<MouseWheel>', on_wheel)
    deadline = time.monotonic()

    def show(slot, frame, trail, color):
        png = base64.b64encode(rgb_png(view.render(frame, trail, color)))
        photos[slot] = tk.PhotoImage(data=png)
        images[slot].configure(image=photos[slot])

    def tick():
        nonlocal deadline, ticks, index, generated, measured
        if not root.winfo_exists():
            return
        now = time.monotonic()
        deadline = max(deadline, now - 3 * period)
        while deadline <= now:
            previous = generated
            generated = generator.step(cmd)
            measured = measured_twist(previous, generated, period)
            plots.append((ticks + 1) * period, cmd.copy(), measured.copy())
            index = (index + 1) % len(reference)
            if index == 0:
                trails[0].clear()
            if ticks % 4 == 0:
                for trail, frame in zip(trails, (reference[index], generated)):
                    trail.append(np.array([frame[0], frame[1], .004]))
            ticks += 1
            deadline += period
        command_text.configure(text=speed_text('指令', cmd))
        measured_text.configure(text=speed_text('机器人', measured))
        if ticks % 2 == 0:
            show(0, reference[index], trails[0], colors[0])
            show(1, generated, trails[1], colors[1])
            plots.draw(chart)
        root.after(max(1, int((deadline - time.monotonic()) * 1000)), tick)

    for name in names:
        name.configure(text=reel.name)
    show(0, reference[index], trails[0], colors[0])
    show(1, generated, trails[1], colors[1])
    root.after(1, tick)
    root.mainloop()
    view.renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_robot_arguments(parser)
    parser.add_argument('--style', type=Path, help='默认用配置里的 default_style')
    parser.add_argument('--checkpoint', type=Path, help='默认用配置里的 checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--speed-step', nargs=3, type=float, metavar=('FORWARD', 'LATERAL', 'TURN'),
                        help='每次点按的增量，单位 m/s、m/s、rad/s')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    spec = resolve_robot(args.robot, args.config)
    speed_step = np.asarray(args.speed_step if args.speed_step is not None else spec.viewer.speed_step)
    if not np.isfinite(speed_step).all() or min(speed_step) <= 0:
        parser.error('--speed-step 需要三个有限的正数')
    torch.set_num_threads(1)
    style = args.style if args.style is not None else spec.default_style
    checkpoint = args.checkpoint if args.checkpoint is not None else spec.checkpoint
    generator = make_generator(
        spec, checkpoint, style, device=args.device,
        seed=spec.train.seed if args.seed is None else args.seed)
    clip = load_clip(style, generator.robot.joints, generator.fps, spec.anchor_body)
    model = display_model(spec.xml)
    heading = clip.yaw[min(generator.cfg.history, len(clip.yaw)) - 1]
    reel = StyleReel(style)
    if FORWARDED_X11:
        view = ForwardedView(model, generator.robot.joints, heading, spec.viewer)
        run_forwarded_viewer(spec.viewer.title, view, generator, reel, speed_step)
        return
    if not glfw.init():
        raise SystemExit('无法初始化 GLFW。请在有图形桌面的会话里打开查看器。')
    try:
        window = glfw.create_window(1200, 780, spec.viewer.title, None, None)
        if not window:
            raise SystemExit('无法创建 MuJoCo OpenGL 窗口。')
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        view = SplitView(model, generator.robot.joints, heading, spec.viewer)
        try:
            run_viewer(window, view, generator, reel, speed_step)
        finally:
            view.context.free()
    finally:
        glfw.terminate()


if __name__ == '__main__':
    main()
