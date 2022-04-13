class Kalmanfilter:
    def __init__(self):  # initialization value
        self.x_P = 0.9
        self.x_Q = 0.08
        self.y_P = 0.9
        self.y_Q = 0.08
        self.z_P = 0.9
        self.z_Q = 0.08
        self.x_priori_estimated_covariance = 1  # prior estimated covariance
        self.y_priori_estimated_covariance = 1
        self.z_priori_estimated_covariance = 1
        # estimated X I need to change number
        self.x_estimated_value_dict = {'hip': 0, 'stomach': 0, 'heart': 0, 'neck': 0, 'nose': 0,
                                       'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
                                       'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
                                       'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # estimated covariance after X
        self.x_post_estimated_covariance_dict = {'hip': 1, 'stomach': 1, 'heart': 1, 'neck': 1, 'nose': 1,
                                                 'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
                                                 'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
                                                 'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
                                                 'left ankle': 1, 'right ankle': 1}
        # estimated Y
        self.y_estimated_value_dict = {'hip': 0, 'stomach': 0, 'heart': 0, 'neck': 0, 'nose': 0,
                                       'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
                                       'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
                                       'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # estimated covariance after Y
        self.y_post_estimated_covariance_dict = {'hip': 1, 'stomach': 1, 'heart': 1, 'neck': 1, 'nose': 1,
                                                 'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
                                                 'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
                                                 'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
                                                 'left ankle': 1, 'right ankle': 1}
                                                 # estimated Y
        self.z_estimated_value_dict = {'hip': 0, 'stomach': 0, 'heart': 0, 'neck': 0, 'nose': 0,
                                       'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
                                       'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
                                       'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # estimated covariance after Y
        self.z_post_estimated_covariance_dict = {'hip': 1, 'stomach': 1, 'heart': 1, 'neck': 1, 'nose': 1,
                                                 'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
                                                 'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
                                                 'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
                                                 'left ankle': 1, 'right ankle': 1}

        # self.x_estimated_value_dict = {'nose': 0, 'left eye': 0, 'right eye': 0, 'left ear': 0, 'right ear': 0,
        #                                'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
        #                                'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
        #                                'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # # estimated covariance after X
        # self.x_post_estimated_covariance_dict = {'nose': 1, 'left eye': 1, 'right eye': 1, 'left ear': 1,
        #                                          'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
        #                                          'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
        #                                          'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
        #                                          'left ankle': 1, 'right ankle': 1}
        # # estimated Y
        # self.y_estimated_value_dict = {'nose': 0, 'left eye': 0, 'right eye': 0, 'left ear': 0, 'right ear': 0,
        #                                'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
        #                                'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
        #                                'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # # estimated covariance after Y
        # self.y_post_estimated_covariance_dict = {'nose': 1, 'left eye': 1, 'right eye': 1, 'left ear': 1,
        #                                          'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
        #                                          'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
        #                                          'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
        #                                          'left ankle': 1, 'right ankle': 1}
        #                                          # estimated Y
        # self.z_estimated_value_dict = {'nose': 0, 'left eye': 0, 'right eye': 0, 'left ear': 0, 'right ear': 0,
        #                                'left shoulder': 0, 'right shoulder': 0, 'left elbow': 0, 'right elbow': 0,
        #                                'left wrist': 0, 'right wrist': 0, 'left hip': 0, 'right hip': 0, 'left knee': 0,
        #                                'right knee': 0, 'left ankle': 0, 'right ankle': 0}
        # # estimated covariance after Y
        # self.z_post_estimated_covariance_dict = {'nose': 1, 'left eye': 1, 'right eye': 1, 'left ear': 1,
        #                                          'right ear': 1, 'left shoulder': 1, 'right shoulder': 1,
        #                                          'left elbow': 1, 'right elbow': 1, 'left wrist': 1, 'right wrist': 1,
        #                                          'left hip': 1, 'right hip': 1, 'left knee': 1, 'right knee': 1,
        #                                          'left ankle': 1, 'right ankle': 1}

    def x_reset(self, P, Q):  # reset P and Q
        self.x_P = P
        self.x_Q = Q

    def y_reset(self, P, Q):  # reset P and Q
        self.y_P = P
        self.y_Q = Q
    
    def z_reset(self, P, Q):  # reset P and Q
        self.z_P = P
        self.z_Q = Q

    def cal_X(self, current_value, label):  # input current value
        self.current_value = current_value
        self.label = label
        self.x_priori_estimated_covariance = self.x_post_estimated_covariance_dict[label]
        x_kalman_gain = self.x_priori_estimated_covariance / (self.x_priori_estimated_covariance + self.x_P)  #
        output = self.x_estimated_value_dict[label] + x_kalman_gain * (
                    current_value - self.x_estimated_value_dict[label])  #
        self.x_estimated_value_dict[label] = output
        self.x_post_estimated_covariance_dict[label] = (
                                                                   1 - x_kalman_gain) * self.x_priori_estimated_covariance + self.x_Q
        self.x_priori_estimated_covariance = self.x_post_estimated_covariance_dict[label]
        return output  # Kalmanfilter formula

    def cal_Y(self, current_value, label):  # input current value
        self.current_value = current_value
        self.label = label
        self.y_priori_estimated_covariance = self.y_post_estimated_covariance_dict[label]
        y_kalman_gain = self.y_priori_estimated_covariance / (self.y_priori_estimated_covariance + self.y_P)  #
        output = self.y_estimated_value_dict[label] + y_kalman_gain * (
                    current_value - self.y_estimated_value_dict[label])  #
        self.y_estimated_value_dict[label] = output
        self.y_post_estimated_covariance_dict[label] = (
                                                                   1 - y_kalman_gain) * self.y_priori_estimated_covariance + self.y_Q
        self.y_priori_estimated_covariance = self.y_post_estimated_covariance_dict[label]
        return output

    def cal_Z(self, current_value, label):  # input current value
        self.current_value = current_value
        self.label = label
        self.z_priori_estimated_covariance = self.z_post_estimated_covariance_dict[label]
        z_kalman_gain = self.z_priori_estimated_covariance / (self.z_priori_estimated_covariance + self.z_P)  #
        output = self.z_estimated_value_dict[label] + z_kalman_gain * (
                    current_value - self.z_estimated_value_dict[label])  #
        self.z_estimated_value_dict[label] = output
        self.z_post_estimated_covariance_dict[label] = (
                                                                   1 - z_kalman_gain) * self.z_priori_estimated_covariance + self.z_Q
        self.z_priori_estimated_covariance = self.z_post_estimated_covariance_dict[label]
        return output


class Point_Kalman_process:
    def __init__(self):
        self.kalman_filter_X = Kalmanfilter()
        self.kalman_filter_Y = Kalmanfilter()
        self.kalman_filter_Z = Kalmanfilter()

    def reset_kalman_filter_X(self, P, Q):
        self.kalman_filter_X.x_reset(P, Q)

    def reset_kalman_filter_Y(self, P, Q):
        self.kalman_filter_Y.y_reset(P, Q)
    
    def reset_kalman_filter_Z(self, P, Q):
        self.kalman_filter_Z.z_reset(P, Q)

    def do_kalman_filter(self, X, Y, Z, label):
        X_cal = self.kalman_filter_X.cal_X(X, label)
        Y_cal = self.kalman_filter_Y.cal_Y(Y, label)
        Z_cal = self.kalman_filter_Z.cal_Z(Z, label)
        return X_cal, Y_cal, Z_cal
