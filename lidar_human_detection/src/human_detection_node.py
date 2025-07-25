#!/usr/bin/env python3

#LiDAR header
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3

import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
import random

# Trajectory hearder
import cv2
import numpy as np
import numpy.linalg as la
from collections import deque

#Data header 
import threading
import time
# from lidar_human_detecion.msg import Num

MAX_POINTS = 10
pointforkal = deque(maxlen=MAX_POINTS)

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        statePre = self.kf.statePre
        statePost = self.kf.statePost
        Q = self.kf.processNoiseCov
        measurementNoiseCov = self.kf.measurementNoiseCov
        errorCovPre = self.kf.errorCovPre
        errorCovPost = self.kf.errorCovPost
        # B = self.kf.controlMatrix
        H = self.kf.measurementMatrix

        # x, y = float(predicted[0]), float(predicted[1])
        x, y = int(predicted[0]), int(predicted[1])

        return (x, y), statePre.T[0], statePost.T[0], errorCovPre #, errorCovPost
        # return predicted[0, 0], predicted[1, 0]
    def kal(self, mu, P, B, u, z):
        A = self.kf.transitionMatrix
        statePre = self.kf.statePre
        # B = self.kf.controlMatrix
        Q = self.kf.processNoiseCov
        # P = self.kf.errorCovPre                     # self.kf.errorCovPost
        H = self.kf.measurementMatrix
        R = self.kf.measurementNoiseCov

        # x_pred = A @ statePre.T[0] + B @ u       
        x_pred = A @ mu + B @ u
        P_pred = A @ P @ A.T + Q / 4
        zp = H @ x_pred

        if z is None:
            return x_pred, P_pred

        epsilon = z - zp

        k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

        x_esti = x_pred + k @ epsilon
        P  = (np.eye(len(P))-k @ H) @ P_pred
        return x_esti, P

# class KalmanFilter:
#     kf = cv2.KalmanFilter(4, 2)
#     kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)                               # : H
#     kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)    # : A


#     def predict(self, coordX, coordY):
#         ''' This function estimates the position of the object'''
#         measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
#         self.kf.correct(measured)
#         predicted = self.kf.predict()       
#         statePre = self.kf.statePre
#         statePost = self.kf.statePost
#         Q = self.kf.processNoiseCov
#         measurementNoiseCov = self.kf.measurementNoiseCov
#         errorCovPre = self.kf.errorCovPre
#         errorCovPost = self.kf.errorCovPost
#         # B = self.kf.controlMatrix
#         H = self.kf.measurementMatrix

#         x, y = float(predicted[0]), float(predicted[1])
#         return (x, y), statePre.T[0], statePost.T[0], errorCovPre #, errorCovPost

#     def kal(self, mu, P, B, u, z):
#         A = self.kf.transitionMatrix
#         statePre = self.kf.statePre
#         # B = self.kf.controlMatrix
#         Q = self.kf.processNoiseCov
#         # P = self.kf.errorCovPre                     # self.kf.errorCovPost
#         H = self.kf.measurementMatrix
#         R = self.kf.measurementNoiseCov

#         # x_pred = A @ statePre.T[0] + B @ u       
#         x_pred = A @ mu + B @ u
#         P_pred = A @ P @ A.T + Q / 4
#         zp = H @ x_pred

#         if z is None:
#             return x_pred, P_pred

#         epsilon = z - zp

#         k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

#         x_esti = x_pred + k @ epsilon
#         P  = (np.eye(len(P))-k @ H) @ P_pred
#         return x_esti, P
    

class CountThread(threading.Thread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.is_shutdown = False

    def run(self):
        while not self.is_shutdown:
            time.sleep(0.1)  # 10ms sleep
            self.node.count += 1
            # self.node.get_logger().info(f"Count value: {self.node.count}")  

    def shutdown(self):
        self.is_shutdown = True

class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        self.subscription = self.create_subscription(PointCloud2, '/velodyne_points', self.pointcloud_callback, 10)
        self.publisher = self.create_publisher(MarkerArray, '/detected_humans', 10)
        self.predpublisher = self.create_publisher(Vector3, '/predicted_humans', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()
        self.count = 0
        self.count_thread = CountThread(self)
        
    def timer_callback(self):
        elapsed_time = time.time() - self.start_time
        count_time_sec = self.count/10
        detected_rate = count_time_sec/elapsed_time*100

        if elapsed_time >= 1.0:
            self.get_logger().info(f'C_t: {count_time_sec:.2f}s, E_t: {elapsed_time:.2f}s, Rate: {detected_rate:.2f}')

    def kalman(x_esti,P,A,Q,B,u,z,H,R):

        x_pred = A @ x_esti + B @ u;         # B : controlMatrix -->  B @ u : gravity% 1.0
        #  x_pred = A @ x_esti or  A @ x_esti - B @ u : upto
        P_pred  = A @ P @ A.T + Q;

        zp = H @ x_pred

        # si no hay observación solo hacemos predicción
        if z is None:
            return x_pred, P_pred, zp

        epsilon = z - zp

        k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

        x_esti = x_pred + k @ epsilon;
        P  = (np.eye(len(P))-k @ H) @ P_pred;
        return x_esti, P, zp    

    def pointcloud_callback(self, msg):
        # self.get_logger().info(f'Received PointCloud2 data with {msg.width * msg.height} points')

        points = []
        for i, data in enumerate(pc2.read_points(msg, skip_nans=True)):
            distance = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)  
            if distance <= 5.0: # 10m 이내의 포인트만 사용
                points.append([data[0], data[1], data[2]])

        points = np.array(points)

        # 클러스터링
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        # self.get_logger().info(f'Found {len(unique_labels) - 1} clusters')

        best_humans = [None, None]
        best_scores = [-1, -1]

        for label in unique_labels:
            if label == -1:  
                continue
            class_member_mask = (labels == label)
            xyz = points[class_member_mask]

            x_min = float(np.min(xyz[:, 0]))
            x_max = float(np.max(xyz[:, 0]))
            y_min = float(np.min(xyz[:, 1]))
            y_max = float(np.max(xyz[:, 1]))
            z_min = float(np.min(xyz[:, 2]))
            z_max = float(np.max(xyz[:, 2]))

            
            if self.is_human(x_min, x_max, y_min, y_max, z_min, z_max):
                score = self.evaluate_cluster(xyz)  
                if score > best_scores[0]:
                    best_scores[1] = best_scores[0]
                    best_humans[1] = best_humans[0]
                    best_scores[0] = score
                    best_humans[0] = (x_min, x_max, y_min, y_max, z_min, z_max, xyz)
                elif score > best_scores[1]:
                    best_scores[1] = score
                    best_humans[1] = (x_min, x_max, y_min, y_max, z_min, z_max, xyz)

        markers = MarkerArray()
        
        id_counter = 0
        if best_humans is not None:
        # for human in best_humans:
            human = best_humans[0]
            if human is not None:
                x_min, x_max, y_min, y_max, z_min, z_max, xyz = human

               
                marker = Marker()
                marker.header.frame_id = "velodyne"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "human_detection"
                marker.id = id_counter
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD

                marker.scale.x = 0.01 

                
                marker.color.a = 1.0  # Transparency
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

               
                points = [
                    (x_min, y_min, z_min), (x_max, y_min, z_min),
                    (x_max, y_min, z_min), (x_max, y_max, z_min),
                    (x_max, y_max, z_min), (x_min, y_max, z_min),
                    (x_min, y_max, z_min), (x_min, y_min, z_min),
                    (x_min, y_min, z_max), (x_max, y_min, z_max),
                    (x_max, y_min, z_max), (x_max, y_max, z_max),
                    (x_max, y_max, z_max), (x_min, y_max, z_max),
                    (x_min, y_max, z_max), (x_min, y_min, z_max),
                    (x_min, y_min, z_min), (x_min, y_min, z_max),
                    (x_max, y_min, z_min), (x_max, y_min, z_max),
                    (x_max, y_max, z_min), (x_max, y_max, z_max),
                    (x_min, y_max, z_min), (x_min, y_max, z_max)
                ]

                for p in points:
                    point = Point()
                    point.x, point.y, point.z = p[0], p[1], p[2]
                    marker.points.append(point)

                markers.markers.append(marker)

                # Text Marker
                center_x = (x_min + x_max) / 2
                center_y = -(y_min + y_max) / 2
                center_z = (z_min + z_max) / 2

                text_marker = Marker()
                text_marker.header.frame_id = "velodyne"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = "human_labels"
                text_marker.id = id_counter + 1000  
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = center_x
                text_marker.pose.position.y = -center_y
                text_marker.pose.position.z = center_z + 1  
                text_marker.pose.orientation.x = 0.0
                text_marker.pose.orientation.y = 0.0
                text_marker.pose.orientation.z = 0.0
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.25  
                text_marker.color.a = 1.0  # Transparency
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
                text_marker.text = f"human{id_counter}\n({center_x:.2f},{center_y:.2f},{center_z:.2f})"

                markers.markers.append(text_marker)

                # pointforkal.append((center_x,center_y))
                x=int(round(center_x, 2)*100)
                y=int(round(center_y, 2)*100)
                pointforkal.append((x,y))

                # pointforkal.append((round(center_x, 2), round(center_y, 2)))
                

                
                kf = KalmanFilter()
                       
                size_of_list = len(pointforkal)
                # print(size_of_list)
                if size_of_list == 10:
                    size=0
                    pmsg = Vector3()
                    for pt in pointforkal:
                        predicted, mu, statePost, errorCovPre = kf.predict(pt[0], pt[1])
                        size+=1
                        if size >= 9:
                            # predicted, mu, statePost, errorCovPre = kf.predict(pt[0], pt[1])
                            pmsg.x=float(predicted[0])-(float(pt[0]-predicted[0])*7)
                            pmsg.y=float(predicted[1])-(float(pt[1]-predicted[1])*7)
                            self.predpublisher.publish(pmsg)
                            # self.get_logger().info(f'{pointforkal}, {round(pmsg.x,2)}, {round(pmsg.y,2)}')
                            # self.get_logger().info(f'{pointforkal}, {pmsg.x}, {pmsg.y}')

                        
                if self.count_thread is None or not self.count_thread.is_alive():
                    self.count_thread = CountThread(self)
                    self.count_thread.start()

                id_counter += 0


        if markers.markers:
            
            self.publisher.publish(markers)

        else:
            
            if self.count_thread is not None and self.count_thread.is_alive():
                self.count_thread.shutdown()
                self.count_thread = None
        
            

    def is_human(self, x_min, x_max, y_min, y_max, z_min, z_max):
        width = x_max - x_min
        depth = y_max - y_min
        height = z_max - z_min

        if 0.2 < width < 1.0 and 0.2 < depth < 1.0 and 0.5 < height < 2.0:
            return True
        return False

    def evaluate_cluster(self, xyz):
        
        return len(xyz)
    
    def update_marker_array(self):
        marker_array = MarkerArray()
        self.publisher.publish(marker_array)
    

def main(args=None):

    rclpy.init(args=args)
    node = HumanDetectionNode()
    try:
        rclpy.spin(node)
    finally:
       
        node.count_thread.shutdown()
        node.count_thread.join()  
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

