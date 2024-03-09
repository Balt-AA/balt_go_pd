import numpy as np

class ControllerNode:
    def __init__(self, thrust_constant, arm_length, moment_constant, input_scaling):
        self.thrust_constant = thrust_constant
        self.arm_length = arm_length
        self.moment_constant = moment_constant

        # Control parameters
        self.zero_position_armed = 100.0
        self.input_scaling = input_scaling

        # Initialize matrices
        self.torques_and_thrust_to_rotor_velocities_ = np.zeros((4, 4))
        self.throttles_to_normalized_torques_and_thrust_ = np.zeros((4, 4))

    def compute_ControlAllocation_and_ActuatorEffect_matrices(self):
        kDegToRad = np.pi / 180.0
        kS = np.sin(45 * kDegToRad)
        rotor_velocities_to_torques_and_thrust = np.array([
                [-kS, kS, kS, -kS],
                [-kS, kS, -kS, kS],
                [-1, -1, 1, 1],
                [1, 1, 1, 1]
                ])
        mixing_matrix = np.array([
                [-0.495384, -0.707107, -0.765306, 1.0],
                [0.495384, 0.707107, -1.0, 1.0],
                [0.495384, -0.707107, 0.765306, 1.0],
                [-0.495384, 0.707107, 1.0, 1.0]
                ])
        
        ## Hardcoded because the calculation of pesudo-inverse is not accurate
        self.throttles_to_normalized_torques_and_thrust_ = np.array([
                [-0.5718, 0.4376, 0.5718, -0.4376],
                [-0.3536, 0.3536, -0.3536, 0.3536],
                [-0.2832, -0.2832, 0.2832, 0.2832],
                [0.2500, 0.2500, 0.2500, 0.2500]
                ])

        # Calculate Control allocation matrix: Wrench to Rotational velocities / k: helper matrix
        k = np.array([self.thrust_constant * self.arm_length,
                      self.thrust_constant * self.arm_length,
                      self.moment_constant * self.thrust_constant,
                      self.thrust_constant])
        rotor_velocities_to_torques_and_thrust *= np.diag(k)
        print("rotor_velocities_to_torques_and_thrust =\n", rotor_velocities_to_torques_and_thrust)

        self.torques_and_thrust_to_rotor_velocities_ = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)
        print("torques_and_thrust_to_rotor_velocities =\n", self.torques_and_thrust_to_rotor_velocities_)
        print("throttles_to_normalized_torques_and_thrust_ =\n", self.throttles_to_normalized_torques_and_thrust_)

    def px4InverseSITL(self, wrench):
        # Initialize vectors
        omega = np.zeros(4)
        throttles = np.zeros(4)
        ones_temp = np.ones((4, 1))

        # Control allocation: Wrench to Rotational velocities (omega)
        omega = np.dot(self.torques_and_thrust_to_rotor_velocities_, wrench)
        omega = np.sqrt(np.abs(omega))  # Element-wise square root, handle negative values
        
        # CBF
        indv_forces = omega * np.abs(omega) * self.thrust_constant
        
        

        throttles = (omega - (self.zero_position_armed * ones_temp.flatten()))
        throttles /= self.input_scaling
        
        # Inverse Mixing: throttles to normalized torques and thrust
        normalized_torque_and_thrust = np.dot(self.throttles_to_normalized_torques_and_thrust_, throttles)

        return normalized_torque_and_thrust, throttles