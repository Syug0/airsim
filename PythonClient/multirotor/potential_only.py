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

def calculate_potential(drone_pos, obstacles, k_att=1.0, k_rep=100.0):
    """ポテンシャル関数を計算する"""
    attractive_potential = k_att * np.linalg.norm(target_pos - drone_pos)
    repulsive_potential = 0
    for obs in obstacles:
        distance = np.linalg.norm(drone_pos - obs)
        if distance < 10.0:  # 10m以内の障害物に対して反発ポテンシャルを計算
            repulsive_potential += k_rep / distance

    total_potential = attractive_potential + repulsive_potential
    return total_potential

def move_towards_target(drone_pos):
    """目標に向かって移動する"""
    direction = target_pos - drone_pos
    if np.linalg.norm(direction) > 0:
        direction /= np.linalg.norm(direction)
    target_position = drone_pos + direction * 5.0  # 5ユニット進む
    client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3).join()

def perform_avoidance(drone_pos):
    """障害物回避動作を実行する"""
    avoidance_direction = np.array([0.0, -1.0, 0.0])  # 左方向に避ける
    avoidance_width = 10.0
    while True:
        # 障害物が近くにある場合は回避方向に移動
        target_position = drone_pos + avoidance_direction * avoidance_width
        client.moveToPositionAsync(target_position[0], target_position[1], drone_pos[2], 3).join()

        # 障害物がまだ検出されるか確認する
        lidar_points = get_lidar_data()
        if not detect_obstacles(lidar_points, 10.0):
            break
        print("障害物が検出されました")
        avoidance_width += 5.0  # 回避幅を増やす

def detect_obstacles(lidar_points, threshold):
    """障害物を検出し、しきい値以内のポイントを返す"""
    if len(lidar_points) == 0:
        return False
    distances = np.linalg.norm(lidar_points[:, :2], axis=1)
    return np.any(distances < threshold)

def main_loop():
    """メインループ"""
    while True:
        drone_state = client.getMultirotorState()
        drone_pos = np.array([drone_state.kinematics_estimated.position.x_val,
                              drone_state.kinematics_estimated.position.y_val,
                              drone_state.kinematics_estimated.position.z_val])

        lidar_points = get_lidar_data()
        if detect_obstacles(lidar_points, 10.0):
            print("障害物が検出されました")
            client.moveByVelocityAsync(0, 0, 0, 1).join()  # ドローンを停止する
            perform_avoidance(drone_pos)
        else:
            move_towards_target(drone_pos)

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
