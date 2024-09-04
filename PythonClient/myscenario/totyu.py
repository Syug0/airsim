import airsim
import numpy as np
import threading
import time

# AirSimに接続し、ドローンを初期化する
client = airsim.MultirotorClient(ip="172.23.32.1", port=41451)
client.confirmConnection()
client.enableApiControl(True, "leader")
client.enableApiControl(True, "follower")
client.armDisarm(True, "leader")
client.armDisarm(True, "follower")
client.takeoffAsync(vehicle_name="leader").join()
client.takeoffAsync(vehicle_name="follower").join()

# スレッドセーフにするためのロックを作成
client_lock = threading.Lock()

# 目標位置をリストとして定義する
target_positions = [
    np.array([18.0, 0.0, -10.0]),  # 前に18ユニット進む
    np.array([18.0, -30.0, -10.0]),  # 左に30ユニット進む
    np.array([50.0, -30.0, -10.0])  # 前に30ユニット進む
]

# 追従機がy+5の位置に到達したかを確認するためのフラグ
follower_ready_event = threading.Event()

def get_lidar_data(vehicle_name):
    """LiDARデータを取得し、3Dポイントクラウドを返す"""
    with client_lock:
        lidarData = client.getLidarData(vehicle_name=vehicle_name, lidar_name="LidarSensor1")
    if len(lidarData.point_cloud) < 3:
        return np.array([])
    point_cloud = np.array(lidarData.point_cloud, dtype=np.float32)
    return point_cloud.reshape(-1, 3)

def get_right_lidar_data(vehicle_name):
    with client_lock:
        lidarData = client.getLidarData(vehicle_name=vehicle_name, lidar_name="LidarSensorRight")
    if len(lidarData.point_cloud) < 3:
         return np.array([])
    point_cloud = np.array(lidarData.point_cloud, dtype=np.float32)
    return point_cloud.reshape(-1, 3)

def calculate_potential(drone_pos, obstacles, k_att=1.0, k_rep=100.0):
    """ポテンシャル関数を計算する"""
    attractive_potential = k_att * np.linalg.norm(target_positions[-1] - drone_pos)
    repulsive_potential = 0
    for obs in obstacles:
        distance = np.linalg.norm(drone_pos - obs)
        if distance < 10.0:  # 10m以内の障害物に対して反発ポテンシャルを計算
            repulsive_potential += k_rep / distance

    total_potential = attractive_potential + repulsive_potential
    return total_potential

def move_leader(target_pos):
    """先導機を目標位置に向かって移動させる"""
    while True:
        with client_lock:
            drone_state = client.getMultirotorState(vehicle_name="leader")
        drone_pos = np.array([drone_state.kinematics_estimated.position.x_val,
                              drone_state.kinematics_estimated.position.y_val,
                              drone_state.kinematics_estimated.position.z_val])
        
        print(f"Leader position: {drone_pos}")
        
        # x軸とy軸の成分のみを使用して方向を計算
        direction = target_pos[:2] - drone_pos[:2]
        distance = np.linalg.norm(direction)
        if distance < 5.0:  # 目標位置に近づいたら停止
            break
        direction /= distance
        target_position = drone_pos + np.array([direction[0], direction[1], 0.0]) * 5.0  # 5ユニット進む
        target_position[2] += 0  # z座標を少しずつ増加させる
        with client_lock:
            client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 3, vehicle_name="leader").join()
        time.sleep(1)  # 1秒待機

def leader_loop():
    """先導機のメインループ"""
    # 追従機がy+5の位置に到達するまで待機
    follower_ready_event.wait()
    
    for target_pos in target_positions:
        move_leader(target_pos)
    print("先導機がすべての目標に到達しました！")

    # 着陸とクリーンアップ
    with client_lock:
        client.landAsync(vehicle_name="leader").join()
        client.armDisarm(False, "leader")
        client.enableApiControl(False, "leader")

def move_towards_target(drone_pos, target_pos, vehicle_name):
    """目標に向かって移動する"""
    direction = target_pos - drone_pos
    if np.linalg.norm(direction) > 0:
        direction /= np.linalg.norm(direction)
    target_position = drone_pos + direction * 5.0  # 5ユニット進む
    with client_lock:
        client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3, vehicle_name=vehicle_name).join()

def perform_formations(drone_pos, vehicle_name):
    while True:
        with client_lock:
            leader_state = client.getMultirotorState(vehicle_name="leader")
            
        leader_pos = np.array([leader_state.kinematics_estimated.position.x_val,
                               leader_state.kinematics_estimated.position.y_val,
                               leader_state.kinematics_estimated.position.z_val])
        
        target_position = leader_pos + np.array([0.0, 5.0, 0.0])
        with client_lock:
                client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3, vehicle_name=vehicle_name).join()
        
        
def perform_avoidance(drone_pos, vehicle_name):
    """障害物回避動作を実行する"""
    avoidance_direction = np.array([0.0, -1.0, 0.0])  # 左方向に避ける
    avoidance_width = 10.0
    while True:
        # 先導機の位置を取得
        with client_lock:
            leader_state = client.getMultirotorState(vehicle_name="leader")
        
        if detect_obstacles(lidar_points, 10.0):
            leader_pos = np.array([drone_pos[0],
                               leader_state.kinematics_estimated.position.y_val,
                               leader_state.kinematics_estimated.position.z_val])
        else:    
            leader_pos = np.array([leader_state.kinematics_estimated.position.x_val,
                                leader_state.kinematics_estimated.position.y_val,
                                leader_state.kinematics_estimated.position.z_val])

        # 追従機と先導機のy座標の距離を計算
        y_distance = abs(drone_pos[1] - leader_pos[1])

        # y座標の距離が5m以上離れている場合は、優先してLeaderのy+5の位置まで移動
        if y_distance > 5.0:
            target_position = leader_pos + np.array([0.0, 5.0, 0.0])
            with client_lock:
                client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3, vehicle_name=vehicle_name).join()
            break

        # 障害物が近くにある場合は回避方向に移動
        target_position = drone_pos + avoidance_direction * avoidance_width

        # 回避の最大値を先導機のy+5で抑える
        if target_position[1] > leader_pos[1] + 5.0:
            target_position[1] = leader_pos[1] + 5.0

        with client_lock:
            client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3, vehicle_name=vehicle_name).join()

        # 障害物がまだ検出されるか確認する
        lidar_points = get_lidar_data(vehicle_name)
        if not detect_obstacles(lidar_points, 10.0):
            break
        print(f"{vehicle_name}：障害物が検出されました")
        # avoidance_width += 5.0  # 回避幅を増やす

def detect_obstacles(lidar_points, threshold):
    """障害物を検出し、しきい値以内のポイントを返す"""
    if len(lidar_points) == 0:
        return False
    distances = np.linalg.norm(lidar_points[:, :2], axis=1)
    return np.any(distances < threshold)

def follower_loop():
    """追従機のメインループ"""
    while True:
        with client_lock:
            leader_state = client.getMultirotorState(vehicle_name="leader")
        leader_pos = np.array([leader_state.kinematics_estimated.position.x_val,
                               leader_state.kinematics_estimated.position.y_val,
                               leader_state.kinematics_estimated.position.z_val])
        
        target_pos = leader_pos + np.array([0.0, 5.0, 0.0])  # 先導機の位置に対してy+5の位置取り

        with client_lock:
            drone_state = client.getMultirotorState(vehicle_name="follower")
        drone_pos = np.array([drone_state.kinematics_estimated.position.x_val,
                              drone_state.kinematics_estimated.position.y_val,
                              drone_state.kinematics_estimated.position.z_val])

        lidar_points = get_lidar_data("follower")
        if detect_obstacles(lidar_points, 10.0):
            print("追従機：障害物が検出されました")
            perform_avoidance(drone_pos, "follower")
        else:
            move_towards_target(drone_pos, target_pos, "follower")
            
        
        right_lidar_points = get_right_lidar_data("follower")
        if not detect_obstacles(lidar_points, 10.0) and detect_obstacles(right_lidar_points, 10.0):
            perform_formations(drone_pos, "follower")
            

        # 目標位置に到達したか確認する
        if np.linalg.norm(drone_pos[:2] - target_pos[:2]) < 1.0:
            print("追従機が目標位置に到達しました！")
            follower_ready_event.set()  # 追従機がy+5の位置に到達したことを通知
            break

    # 追従機が目標位置に到達した後も先導機を追従し続ける
    while True:
        with client_lock:
            leader_state = client.getMultirotorState(vehicle_name="leader")
        leader_pos = np.array([leader_state.kinematics_estimated.position.x_val,
                               leader_state.kinematics_estimated.position.y_val,
                               leader_state.kinematics_estimated.position.z_val])
        
        target_pos = leader_pos + np.array([0.0, 5.0, 0.0])  # 先導機の位置に対してy+5の位置取り

        with client_lock:
            drone_state = client.getMultirotorState(vehicle_name="follower")
        drone_pos = np.array([drone_state.kinematics_estimated.position.x_val,
                              drone_state.kinematics_estimated.position.y_val,
                              drone_state.kinematics_estimated.position.z_val])

        lidar_points = get_lidar_data("follower")
        if detect_obstacles(lidar_points, 10.0):
            print("追従機：障害物が検出されました")
            perform_avoidance(drone_pos, "follower")
            with client_lock:
                client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="follower").join()  # ドローンを停止する
            perform_avoidance(drone_pos, "follower")
        else:
            move_towards_target(drone_pos, target_pos, "follower")

        time.sleep(0.3)  # 0.3秒待機

# 両方のメインループを並行して実行する
leader_thread = threading.Thread(target=leader_loop)
follower_thread = threading.Thread(target=follower_loop)

leader_thread.start()
follower_thread.start()

leader_thread.join()
follower_thread.join()