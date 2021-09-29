from pybullet_multigoal_gym.robots.robot_bases import URDFBasedRobot
from gym import spaces
import numpy as np


class Kuka(URDFBasedRobot):
    def __init__(self, gripper_type='parallel_jaw', joint_control=False, grasping=False, end_effector_start_on_table=False,
                 obj_range=0.15, target_range=0.15):
        self.gripper_type = gripper_type
        if self.gripper_type == 'robotiq85':
            model_urdf = 'robots/kuka/iiwa14_robotiq85.urdf'
        else:
            model_urdf = 'robots/kuka/iiwa14_parallel_jaw.urdf'
        URDFBasedRobot.__init__(self,
                                model_urdf=model_urdf,
                                robot_name='iiwa14',
                                self_collision=False,
                                fixed_base=True)
        self.kuka_body_index = None
        self.kuka_joint_index = None
        # initial robot joint states
        self.kuka_rest_pose = [0, -0.5592432, 0, 1.733180, 0, -0.8501557, 0]
        self.joint_state_target = None
        self.end_effector_tip_joint_index = None
        self.end_effector_target = None
        self.end_effector_tip_initial_position = np.array([-0.52, 0.0, 0.25])
        if end_effector_start_on_table:
            self.end_effector_tip_initial_position[-1] = 0.175

        self.end_effector_xyz_upper = np.array([-0.37, 0.20, 0.55])
        self.end_effector_xyz_lower = np.array([-0.67, -0.20, 0.175])
        self.end_effector_fixed_quaternion = [0, -1, 0, 0]
        self.object_bound_lower = self.end_effector_tip_initial_position.copy() - obj_range
        self.object_bound_lower[0] += 0.03
        self.object_bound_upper = self.end_effector_tip_initial_position.copy() + obj_range
        self.object_bound_upper[0] -= 0.03
        self.target_bound_lower = self.end_effector_tip_initial_position.copy() - target_range
        self.target_bound_lower[0] += 0.03
        self.target_bound_lower[-1] = self.end_effector_xyz_lower[-1]
        self.target_bound_upper = self.end_effector_tip_initial_position.copy() + target_range
        self.target_bound_upper[0] -= 0.03

        self.gripper_joint_index = None
        if self.gripper_type == 'robotiq85':
            self.gripper_joint_name = [
                'iiwa_gripper_finger1_joint',
                'iiwa_gripper_finger2_joint',
                'iiwa_gripper_finger1_inner_knuckle_joint',
                'iiwa_gripper_finger1_finger_tip_joint',
                'iiwa_gripper_finger2_inner_knuckle_joint',
                'iiwa_gripper_finger2_finger_tip_joint'
            ]
            self.gripper_abs_joint_limit = 0.804
            self.gripper_grasp_block_state = 0.545
            self.gripper_mmic_joint_multiplier = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
        else:
            self.gripper_joint_name = [
                'iiwa_gripper_finger1_joint',
                'iiwa_gripper_finger2_joint'
            ]
            self.gripper_abs_joint_limit = 0.035
            self.gripper_grasp_block_state = 0.02
            self.gripper_mmic_joint_multiplier = np.array([1.0, 1.0])
        self.gripper_num_joint = len(self.gripper_joint_name)
        self.gripper_tip_offset = 0.0

        # action space
        self.joint_control = joint_control
        self.grasping = grasping
        if self.joint_control:
            if self.grasping:
                self.action_space = spaces.Box(-np.ones([8]), np.ones([8]))
            else:
                self.action_space = spaces.Box(-np.ones([7]), np.ones([7]))
        else:
            if self.grasping:
                self.action_space = spaces.Box(-np.ones([4]), np.ones([4]))
            else:
                self.action_space = spaces.Box(-np.ones([3]), np.ones([3]))

    def robot_specific_reset(self, bullet_client):
        if self.kuka_body_index is None:
            self.kuka_body_index = self.jdict['plane_iiwa_joint'].bodies[self.jdict['plane_iiwa_joint'].bodyIndex]
        if self.kuka_joint_index is None:
            # The 0-th joint is the one that connects the world frame and the kuka base, so skip it
            self.kuka_joint_index = [
                self.jdict['iiwa_joint_1'].jointIndex,
                self.jdict['iiwa_joint_2'].jointIndex,
                self.jdict['iiwa_joint_3'].jointIndex,
                self.jdict['iiwa_joint_4'].jointIndex,
                self.jdict['iiwa_joint_5'].jointIndex,
                self.jdict['iiwa_joint_6'].jointIndex,
                self.jdict['iiwa_joint_7'].jointIndex,
            ]
        if self.end_effector_tip_joint_index is None:
            self.end_effector_tip_joint_index = self.jdict['iiwa_gripper_tip_joint'].jointIndex
        if self.gripper_joint_index is None:
            if self.gripper_type == 'robotiq85':
                self.gripper_joint_index = [
                    self.jdict['iiwa_gripper_finger1_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger1_inner_knuckle_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger1_finger_tip_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_inner_knuckle_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_finger_tip_joint'].jointIndex,
                ]
            else:
                self.gripper_joint_index = [
                    self.jdict['iiwa_gripper_finger1_joint'].jointIndex,
                    self.jdict['iiwa_gripper_finger2_joint'].jointIndex,
                ]
        # reset arm poses
        self.set_kuka_joint_state(self.kuka_rest_pose)
        self.kuka_rest_pose = self.compute_ik(bullet_client, self.end_effector_tip_initial_position)
        self.set_kuka_joint_state(self.kuka_rest_pose)
        self.set_finger_joint_state(self.gripper_abs_joint_limit)
        self.move_finger(bullet_client=bullet_client, grip_ctrl=self.gripper_abs_joint_limit)
        self.end_effector_target = self.parts['iiwa_gripper_tip'].get_position()
        self.joint_state_target, _ = self.get_kuka_joint_state()

    def apply_action(self, a, bullet_client):

        if self.grasping:
            # map action in [-1, 1] to gripper joint range
            grip_ctrl = (a[-1] + 1.0) * (self.gripper_abs_joint_limit / 2)
            self.move_finger(bullet_client=bullet_client,
                             grip_ctrl=grip_ctrl)

        if self.joint_control:
            self.joint_state_target = (a[:7] * 0.05) + self.joint_state_target
            joint_poses = self.joint_state_target.copy()
        elif len(a) == 7:
            joint_poses = self.compute_ik(bullet_client=bullet_client,
                                          target_ee_pos=a[:3],
                                          target_ee_quat=np.array([0, -1, 0, 0]),
                                          )
        else:
            # actions alter the ee target pose
            self.end_effector_target += (a[:3] * 0.05)
            self.end_effector_target = np.clip(self.end_effector_target,
                                               self.end_effector_xyz_lower,
                                               self.end_effector_xyz_upper)
            joint_poses = self.compute_ik(bullet_client=bullet_client,
                                          target_ee_pos=self.end_effector_target)

        self.move_arm(bullet_client=bullet_client, joint_poses=joint_poses)
        for _ in range(5):
            # ensure the action is finished
            bullet_client.stepSimulation()

    def calc_robot_state(self):
        # gripper tip states in the world frame
        gripper_xyz = self.parts['iiwa_gripper_tip'].get_position()
        gripper_rpy = self.parts['iiwa_gripper_tip'].get_orientation_eular()
        gripper_vel_xyz = self.parts['iiwa_gripper_tip'].get_linear_velocity()
        gripper_vel_rpy = self.parts['iiwa_gripper_tip'].get_angular_velocity()
        if self.grasping:
            # calculate distance between the gripper finger tabs
            gripper_finger1_tab_xyz = np.array(self.parts['iiwa_gripper_finger1_finger_tab_link'].get_position())
            gripper_finger2_tab_xyz = np.array(self.parts['iiwa_gripper_finger2_finger_tab_link'].get_position())
            gripper_finger_closeness = np.sqrt(
                np.sum(np.square(gripper_finger1_tab_xyz - gripper_finger2_tab_xyz))).ravel()
            # calculate finger joint velocity instead of using a get() method due to compatibility among grippers
            grip_base_vel = self.parts['iiwa_gripper_base_link'].get_linear_velocity()
            grip_finger_vel = self.parts['iiwa_gripper_finger1_finger_tab_link'].get_linear_velocity()
            gripper_finger_vel = (grip_base_vel - grip_finger_vel)[1].ravel()
        else:
            # symmetric gripper
            gripper_finger_closeness = np.array([0.0])
            gripper_finger_vel = np.array([0.0])
        if self.joint_control:
            joint_poses, _ = self.get_kuka_joint_state()
        else:
            joint_poses = None
        return gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses

    def compute_ik(self, bullet_client, target_ee_pos, target_ee_quat=None):
        assert target_ee_pos.shape == (3,)
        if target_ee_quat is None:
            target_ee_quat = self.end_effector_fixed_quaternion
        else:
            assert target_ee_quat.shape == (4,)
        # kuka-specific values for ik computation using null space dumping method,
        #   obtained from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        joint_poses = bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.kuka_body_index,
            endEffectorLinkIndex=self.end_effector_tip_joint_index,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_quat,
            # lower limits for null space
            lowerLimits=[-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],
            # upper limits for null space
            upperLimits=[.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],
            # joint ranges for null space
            jointRanges=[5.8, 4, 5.8, 4, 5.8, 4, 6],
            restPoses=self.kuka_rest_pose,
            maxNumIterations=40,
            residualThreshold=0.00001)
        return joint_poses[:7]

    def move_arm(self, bullet_client, joint_poses):
        bullet_client.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                                jointIndices=self.kuka_joint_index,
                                                controlMode=bullet_client.POSITION_CONTROL,
                                                targetPositions=joint_poses,
                                                targetVelocities=np.zeros((7,)),
                                                forces=np.ones((7,)) * 15,
                                                positionGains=np.ones((7,)) * 0.03,
                                                velocityGains=np.ones((7,)))

    def move_finger(self, bullet_client, grip_ctrl):
        target_joint_poses = self.gripper_mmic_joint_multiplier * grip_ctrl
        bullet_client.setJointMotorControlArray(bodyUniqueId=self.kuka_body_index,
                                                jointIndices=self.gripper_joint_index,
                                                controlMode=bullet_client.POSITION_CONTROL,
                                                targetPositions=target_joint_poses,
                                                targetVelocities=np.zeros((self.gripper_num_joint,)),
                                                forces=np.ones((self.gripper_num_joint,)) * 50,
                                                positionGains=np.ones((self.gripper_num_joint,)) * 0.03,
                                                velocityGains=np.ones((self.gripper_num_joint,)))

    def get_kuka_joint_state(self):
        kuka_joint_pos = []
        kuka_joint_vel = []
        for i in range(len(self.kuka_joint_index)):
            x, vx = self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].get_state()
            kuka_joint_pos.append(x)
            kuka_joint_vel.append(vx)
        return kuka_joint_pos, kuka_joint_vel

    def set_kuka_joint_state(self, pos=None, vel=None, gripper_tip_pos=None, bullet_client=None):
        if gripper_tip_pos is not None:
            assert bullet_client is not None
            pos = self.compute_ik(bullet_client=bullet_client,
                                  target_ee_pos=gripper_tip_pos)
        pos = np.array(pos)
        if vel is None:
            vel = np.zeros(pos.shape[0])
        for i in range(len(pos)):
            self.jdict['iiwa_joint_' + str(self.kuka_joint_index[i])].reset_position(pos[i], vel[i])

    def get_finger_joint_state(self):
        finger_joint_pos = []
        finger_joint_vel = []
        for name in self.gripper_joint_name:
            x, vx = self.jdict[name].get_state()
            finger_joint_pos.append(x)
            finger_joint_vel.append(vx)
        return finger_joint_pos, finger_joint_vel

    def set_finger_joint_state(self, pos, vel=None):
        pos = pos * self.gripper_mmic_joint_multiplier
        if vel is None:
            vel = np.zeros(pos.shape[0])
        for i in range(pos.shape[0]):
            self.jdict[self.gripper_joint_name[i]].reset_position(pos[i], vel[i])
