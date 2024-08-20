import airsim
import numpy as np

# AirSimに接続し、ドローンを初期化する
client = airsim.MultirotorClient(ip="172.23.32.1", port=41451)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# 目標位置を定義する
target_pos = np.array([50.0, 0.0, -10.0])

def get_lidar_data():
    """LiDARデータを取得し、3Dポイントクラウドを返す"""
    lidarData = client.getLidarData()  # 引数なしで呼び出し
    if len(lidarData.point_cloud) < 3:
        return np.array([])
    point_cloud = np.array(lidarData.point_cloud, dtype=np.float32)
    return point_cloud.reshape(-1, 3)

def calculate_potential_gradient(drone_pos, obstacles, k_att=1.0, k_rep=100.0):
    """ポテンシャル場の勾配を計算する"""
    # 吸引力（目標に向かう力）の計算
    direction_to_target = target_pos - drone_pos
    distance_to_target = np.linalg.norm(direction_to_target)
    if distance_to_target > 0:
        attractive_force = k_att * direction_to_target / distance_to_target
    else:
        attractive_force = np.array([0.0, 0.0, 0.0])

    # 反発力（障害物を避ける力）の計算
    repulsive_force = np.array([0.0, 0.0, 0.0])
    for obs in obstacles:
        direction_to_obs = drone_pos - obs
        distance_to_obs = np.linalg.norm(direction_to_obs)
        if distance_to_obs < 10.0:  # 10m以内の障害物に対して反発力を計算
            if distance_to_obs > 0:
                repulsive_force += k_rep * direction_to_obs / (distance_to_obs ** 2)
    
    # 合成力を返す
    total_force = attractive_force + repulsive_force
    return total_force

def move_drone(drone_pos, force_vector):
    """ドローンを力ベクトルに基づいて移動する"""
    if np.linalg.norm(force_vector) > 0:
        force_vector /= np.linalg.norm(force_vector)
    target_position = drone_pos + force_vector * 5.0  # 5ユニット進む
    client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 3).join()

def main_loop():
    """メインループ"""
    while True:
        drone_state = client.getMultirotorState()
        drone_pos = np.array([drone_state.kinematics_estimated.position.x_val,
                              drone_state.kinematics_estimated.position.y_val,
                              drone_state.kinematics_estimated.position.z_val])

        lidar_points = get_lidar_data()

        # ポテンシャル場の勾配に基づいて力を計算し、ドローンを動かす
        force_vector = calculate_potential_gradient(drone_pos, lidar_points)
        move_drone(drone_pos, force_vector)

        # 目標位置に到達したか確認する
        if np.linalg.norm(drone_pos[:2] - target_pos[:2]) < 1.0:
            print("目標に到達しました！")
            break

    # 着陸とクリーンアップ
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

# メインループを実行する
main_loop()
