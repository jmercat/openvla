"""This code is adapted from Anikait Singh's code: https://github.com/Asap7772/rt1_eval/blob/master/kitchen_eval/widowx_real_env.py"""
import numpy as np
import os
from widowx_envs.utils.exceptions import Environment_Exception
from widowx_envs.widowx.widowx_env import WidowXEnv
from gym.spaces import Box
from gym.spaces import Dict


class JaxRLWidowXEnv(WidowXEnv):
    def __init__(self, env_params=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0 # used for reward function
                 ):

        super().__init__(env_params)
        self.image_size = fixed_image_size
        self.image_height, self.image_width = 480, 640
        self.task_id = task_id
        self.num_tasks = num_tasks

        obs_dict = {}
        if not self._hp.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        if self._hp.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if self._hp.add_task_id:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.move_except = False
        self.control_viewpoint = control_viewpoint
        self.spec = None
        self.requires_timed = True
        self.do_render = True
        self.traj_counter = 0

    def _default_hparams(self):
        from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
        default_dict = {
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'camera_topics': [IMTopic('/camera0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'add_states': False,
            'add_task_id':False,
            'from_states': False,
            'reward_type': None
        }
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None, reset_state=None):
        if itraj is None:
            itraj = self.traj_counter
        self.traj_counter += 1
        return super().reset(itraj, reset_state)

    def _get_processed_image(self, image=None):
        from skimage.transform import resize
        downsampled_trimmed_image = resize(image, (self.image_height, self.image_width), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        return downsampled_trimmed_image

    def step(self, action):
        obs = super().step(action['action'].squeeze(), action['tstamp_return_obs'], blocking=False)
        reward = 0
        done = obs['full_obs']['env_done']  # done can come from VR buttons
        info = {}
        if self.move_except:
            done = True
            info['Error.truncated'] = True
        return obs, reward, done, info

    def disable_render(self):
        self.do_render = False

    def enable_render(self):
        self.do_render = True

    def _get_obs(self):
        full_obs = super()._get_obs()
        obs = {}
        if self.do_render:
            processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
            obs['pixels'] = processed_images
        obs['full_obs'] = full_obs
        if self._hp.add_states:
            obs['state'] = self.get_full_state()[None]
        return obs

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id_vec(self, task_id):
        task_id_vec = None
        if (task_id is not None) and self.num_tasks:
            task_id_vec = np.zeros(self.num_tasks, dtype=np.float32)[None]
            task_id_vec[:, task_id] = 1.0
        return task_id_vec

    def move_to_startstate(self, start_state=None):
        successful = False
        print('entering move to startstate loop.')
        while not successful:
            try:
                # Get XYZ position from user.
                x_val = input('Enter x value of gripper starting pos (default = 0.30): ')
                if x_val == '':
                    x_val = 0.30
                y_val = input('Enter y value of gripper starting pos (default = -0.09): ')
                if y_val == '':
                    y_val = -0.09
                z_val = input('Enter z value of gripper starting pos (default = 0.25): ')
                if z_val == '':
                    z_val = 0.25
                # Fix initial orientation & add user's commanded XYZ into start transform.
                transform = np.array([
                    [0.22, -0.04, 0.97, float(x_val)],
                    [-0.01, 1.00, 0.05, float(y_val)],
                    [-0.98, -0.02, 0.22, float(z_val)],
                    [0.00, 0.00, 0.00, 1.00],
                ])
                self.controller.move_to_starteep(transform, duration=0.8)
                successful = True
            except Environment_Exception:
                self.move_to_neutral()
