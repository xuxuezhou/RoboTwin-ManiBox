
from .base_task import Base_task
from .utils import *
import sapien
import math

class bowls_stack(Base_task):

    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot(**kwags)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()
        self.pre_move()
        self.robot.set_origin_endpose()
        self.load_actors()
        
        self.step_lim = 900

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        bowl_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.767],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False
        )

        while abs(bowl_pose.p[0]) < 0.05 or np.sum(pow(bowl_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0169:
            bowl_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.767],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False
            )

        self.bowl1, self.bowl1_data = create_actor(
            self.scene,
            pose=bowl_pose,
            modelname="002_container_test",
            model_id=4,
            z_val_protect=True,
            convex=True
        )

        bowl_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.767],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False
        )

        while abs(bowl_pose.p[0]) < 0.05 or np.sum(pow(bowl_pose.p[:2] - self.bowl1.get_pose().p[:2],2)) < 0.0169 \
              or np.sum(pow(bowl_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0169:
            bowl_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.767],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False
            )


        self.bowl2, self.bowl2_data = create_actor(
            self.scene,
            pose=bowl_pose,
            modelname="002_container_test",
            model_id=4,
            z_val_protect=True,
            convex=True
        )

        bowl_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.767],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False
        )

        while abs(bowl_pose.p[0]) < 0.05 or np.sum(pow(bowl_pose.p[:2] - self.bowl1.get_pose().p[:2],2)) < 0.0169 or\
              np.sum(pow(bowl_pose.p[:2] - self.bowl2.get_pose().p[:2],2)) < 0.0169 \
              or np.sum(pow(bowl_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0169:
            bowl_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.767],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False
            )


        self.bowl3, self.bowl3_data = create_actor(
            self.scene,
            pose=bowl_pose,
            modelname="002_container_test",
            model_id=4,
            z_val_protect=True,
            convex=True
        )

        self.bowl1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.bowl2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.bowl3.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        pose = self.bowl1.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])
        pose = self.bowl2.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04])  
        pose = self.bowl3.get_pose().p
        self.prohibited_area.append([pose[0]-0.04,pose[1]-0.04,pose[0]+0.04,pose[1]+0.04]) 
        target_pose = [-0.04,-0.13,0.04,-0.05]
        self.prohibited_area.append(target_pose)

    def move_bowl(self,actor,actor_data,target_pose, id = 0, las_arm = None):
        actor_pose = actor.get_pose().p
        if actor_pose[0] >0:
            now_arm = 'right'
            pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor, actor_data, pre_dis=0.1, contact_point_id=0)
            target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor, actor_data, pre_dis=0., contact_point_id=0)
            if now_arm == las_arm or las_arm is None:
                    self.right_move_to_pose(pre_grasp_pose)
            else:
                self.together_move_to_pose(left_target_pose=self.robot.left_original_pose,right_target_pose=pre_grasp_pose)
            self.right_move_to_pose(pose = target_grasp_pose)
            self.close_right_gripper()
            self.right_move_to_pose(pose = pre_grasp_pose)
            place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor, actor_data, endpose_tag=now_arm, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0.09)
            target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor, actor_data, endpose_tag=now_arm, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0)
            self.right_move_to_pose(place_pose)
            self.right_move_to_pose(target_place_pose)
            self.open_right_gripper()
            self.right_move_to_pose(place_pose)
        else:
            now_arm = 'left'
            pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor, actor_data, pre_dis=0.1, contact_point_id=2)
            target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor, actor_data, pre_dis=0., contact_point_id=2)
            if now_arm == las_arm or las_arm is None:
                    self.left_move_to_pose(pre_grasp_pose)
            else:
                self.together_move_to_pose(left_target_pose=pre_grasp_pose,right_target_pose=self.robot.right_original_pose)
            self.left_move_to_pose(pose = target_grasp_pose)
            self.close_left_gripper()
            self.left_move_to_pose(pose = pre_grasp_pose)
            place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor, actor_data, endpose_tag=now_arm, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0.09)
            target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor, actor_data, endpose_tag=now_arm, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=[0,0.707,0.707,0], pre_dis=0)
            self.left_move_to_pose(place_pose)
            self.left_move_to_pose(target_place_pose)
            self.open_left_gripper()
            self.left_move_to_pose(place_pose)
        return now_arm
    
    def play_once(self):
        las_arm = self.move_bowl(self.bowl1, self.bowl1_data, [0, -0.1, 0.75], id = 0, las_arm = None)
        las_arm = self.move_bowl(self.bowl2, self.bowl2_data, self.bowl1.get_pose().p + [0,0,0.03], id = 1, las_arm = las_arm)
        las_arm = self.move_bowl(self.bowl3, self.bowl3_data, self.bowl2.get_pose().p + [0,0,0.03], id = 2, las_arm = las_arm)
            
        info = dict()
        info['messy_table_info'] = self.record_messy_objects
        info['texture_info'] = {'wall_texture': self.wall_texture, 'table_texture': self.table_texture}
        return info
        
    def stage_reward(self):
        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        bowl3_pose = self.bowl3.get_pose().p
        bowl1_pose, bowl2_pose, bowl3_pose = sorted([bowl1_pose, bowl2_pose, bowl3_pose],key = lambda x:x[2])
        target_pose = [0,-0.1]
        target_height = [0.766, 0.798, 0.829]
        eps = 0.02        
        reward = 0
        if np.all(abs(bowl1_pose[:2] - bowl2_pose[:2]) < eps) and \
           np.all(abs(bowl3_pose[:2] - bowl2_pose[:2]) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open():
            reward += 0.3
            if np.all(np.array([bowl1_pose[2], bowl2_pose[2], bowl3_pose[2]]) - target_height < eps) and self.is_left_gripper_open() and self.is_right_gripper_open():
                reward += 0.7
        return reward
    
    def check_success(self):
        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        bowl3_pose = self.bowl3.get_pose().p
        bowl1_pose, bowl2_pose, bowl3_pose = sorted([bowl1_pose, bowl2_pose, bowl3_pose],key = lambda x:x[2])
        target_pose = [0,-0.1]
        target_height = [0.766, 0.798, 0.829]
        eps = 0.02        
        return np.all(abs(bowl1_pose[:2] - bowl2_pose[:2]) < eps) and \
               np.all(abs(bowl3_pose[:2] - bowl2_pose[:2]) < eps) and \
               np.all(np.array([bowl1_pose[2], bowl2_pose[2], bowl3_pose[2]]) - target_height < eps) and self.is_left_gripper_open() and self.is_right_gripper_open()