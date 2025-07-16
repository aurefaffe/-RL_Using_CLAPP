import math
import argparse
import numpy as np

import miniworld
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, ImageFrame, MeshEnt, Key
from miniworld.params import DEFAULT_PARAMS
from miniworld.manual_control import ManualControl
from typing import Tuple, Optional, Union, Dict, Any
from gymnasium.core import ObsType


from gymnasium import utils, spaces
import gymnasium as gym

class MyTmaze(MiniWorldEnv, utils.EzPickle):

    def __init__(self, add_obstacles = False, add_visual_cue_object = False, intermediate_rewards = False,reward_left = True,
                 probability_of_left = 0.0,latent_learning = False, add_visual_cue_image = False, left_arm = True, right_arm = True, **kwargs):



        DEFAULT_PARAMS.set('forward_step',0.15*5, 0.12*5, 5*0.17)
        DEFAULT_PARAMS.set("wall_tex", "picket_fence")
        DEFAULT_PARAMS.set("turn_step", 30)



        self.latent_learning = latent_learning
        self.intermediate_rewards = intermediate_rewards
        self.add_obstacles = add_obstacles
        self.reward_left = reward_left
        self.add_visual_cue_object = add_visual_cue_object
        self.add_visual_cue_image = add_visual_cue_image
        self.probability_of_left = probability_of_left
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.agent = None


        if self.add_obstacles:
            self.num_obstacles = 3


        MiniWorldEnv.__init__(self, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def place_agent(self, room=None, pos=None, dir=None, min_x=None, max_x=None, min_z=None, max_z=None):
        return super().place_agent(room, pos, dir, min_x, max_x, min_z, max_z)


    def _gen_world(self):

        min_z_room2 = -6.85 if self.left_arm else -1.37
        max_z_room2 = 6.85 if self.right_arm else 1.37

        room1 = self.add_rect_room(min_x=-0.22, max_x=8, min_z=-1.37, max_z=1.37, wall_tex="picket_fence")
        room2 = self.add_rect_room(min_x=8, max_x=10.74, min_z= min_z_room2, max_z= max_z_room2, wall_tex="picket_fence")

        self.connect_rooms(room_a= room1, room_b= room2, min_z= -1.37, max_z= 1.37)

        if not self.latent_learning:
            self.box = Box(color="red", size=1)
            # self.box2 = Box(color="red", size=1)
            self.key = Key(color= 'red')
            self.found_key = False


            if self.reward_left or self.np_random.uniform() < self.probability_of_left:
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 1)
                # self.place_entity(self.box2, room=room2, min_z=-0.61, max_z=-0.27, min_x=9.5, max_x=9.5)
                self.found_box2 = False
                if self.add_visual_cue_object:
                    self.place_entity(ent = self.key, room = room2, min_z= -1.37 ,max_z=  -0.5)
                self.reward_left = True

            else:
                self.place_entity(self.box, room=room2, min_z= room2.max_z - 1)
                if self.add_visual_cue_object:
                    self.place_entity(ent = self.key, room = room2, min_z= 0.5, max_z= 1.37)


        if self.add_obstacles:
            for i in range(self.num_obstacles):
                self.place_entity(
                    ent = MeshEnt(mesh_name= 'barrel.obj',height= 1)
                    )



        #self.place_agent(room= room1, dir=self.np_random.uniform(-math.pi / 4, math.pi / 4))
        self.agent=self.place_agent(pos= [0.20,0,-0.90], dir= 0)

        pos_list = [[1.37*(2*x+1)-0.22, 1.37, -1.37] for x in range(3)] \
                + [[1.37*(2*x+1)-0.22, 1.37, 1.37] for x in range(3)] \
                + [[10.74, 1.37, 1.37*(2*x+1)-6.85] for x in range(5)] \
                + [[8, 1.37, 1.37*(2*x+1)-6.85] for x in [0, 1, 3, 4]] \
                + [[9.37, 1.37, -6.85], [9.37, 1.37, 6.85]]

        dir_list = [-math.pi / 2 for _ in range(3)] \
                + [math.pi / 2 for _ in range(3)] \
                + [-math.pi for _ in range(5)] \
                + [0 for _ in range(4)] \
                + [-math.pi / 2 , math.pi / 2]

        # append images to proper positions with desried direction
        for i, (pos_, dir_) in enumerate(zip(pos_list, dir_list)):
            self.entities.append(
                ImageFrame(
                    pos=pos_, dir=dir_, width=2.74, tex_name="stl{}".format(i + 13 if self.reward_left and i ==  8 else i )
                )
            )
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
            """
            Reset l'environnement en appelant le parent et en ajoutant agent_pos/agent_dir.
            Returns:
                Tuple[ObsType, dict]: 
                    - obs: l'observation (retournée par le parent)
                    - info: le dictionnaire du parent + agent_pos et agent_dir
            """
            # Appel à la méthode parente et récupération de obs et info
            obs, info = super().reset(seed=seed, options=options)  # type: ignore

            # Ajout des infos de l'agent au dictionnaire info
            info["agent_pos"] = self.agent.pos  # Supposant que self.agent.pos existe
            info["agent_dir"] = self.agent.dir  # Supposant que self.agent.dir existe

            return obs, info        


    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True
        # if self.near(self.box2) and not self.found_box2:
        #     reward += self._reward()*0.4
        #     self.entities.remove(self.box2)
        #     self.found_box2 = True

        if self.add_visual_cue_object and self.found_key == False and self.near(self.key):
           self.found_key = True
           if self.intermediate_rewards:
               reward += self._reward()
           self.entities.remove(self.key)

        info["goal_pos"] = self.box.pos
        info["agent_pos"] = self.agent.pos
        info["agent_dir"] = self.agent.dir

        return obs, reward, termination, truncation, info





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MyTMaze", help="name of the environment")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = ManualControl(env, args.no_time_limit, args.domain_rand)
    manual_control.run()


if __name__ == "__main__":
    # make sure register the environment before running
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='custom_T_Maze_V0:MyTmaze',
    )
    main()
