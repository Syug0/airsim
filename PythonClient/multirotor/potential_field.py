import airsim
import numpy as np
import time

# 定数の設定
GOAL_POS = np.array([50.0, 0.0, -10.0])  # ゴール地点
K_ATTRACT = 1.0  # アトラクティブフィールドの係数
K_REPEL = 100.0  # リパルシブフィールドの係数
REPEL_RANGE = 5.0  # 斥力の有効範囲
SAFE_DISTANCE = 5.0  # 安全距離

def compute_potential_field(current_pos, orientation, obstacles):
    # アトラクティブフィールドの計算
    attractive_force = K_ATTRACT * (GOAL_POS - current_pos)
    
    # ドローンの前方方向ベクトルを計算
    forward_vector = np.array([1.0, 0.0, 0.0])  # ドローンの前方方向（x方向）

    # リパルシブフィールドの計算
    repulsive_force = np.array([0.0, 0.0, 0.0])
    for obs in obstacles:
        obs_vec = current_pos - obs
        distance = np.linalg.norm(obs_vec)
        if distance < SAFE_DISTANCE:
            # 障害物までのベクトルを正規化
            obs_vec_normalized = obs_vec / distance
            # 前方方向に対する障害物の位置を相対的に計算
            side_direction = np.cross(forward_vector, np.array([0, 0, 1]))
            if np.dot(obs_vec_normalized, side_direction) > 0:
                direction = np.cross(obs_vec, np.array([0, 0, 1]))  # 横方向の回避力
                direction = direction / np.linalg.norm(direction)  # 単位ベクトル化
                repulsive_force += K_REPEL * (1.0 / distance - 1.0 / REPEL_RANGE) * direction / distance**2

    # 合成フィールドの計算
    total_force = attractive_force + repulsive_force
    return total_force

# AirSimクライアントの設定
client = airsim.MultirotorClient("172.23.32.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

try:
    while True:
        # 現在のドローン位置と姿勢を取得
        state = client.getMultirotorState()
        current_pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        orientation = np.array([
            state.kinematics_estimated.orientation.x_val,
            state.kinematics_estimated.orientation.y_val,
            state.kinematics_estimated.orientation.z_val,
            state.kinematics_estimated.orientation.w_val
        ])

        # LIDARセンサデータを取得
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            print("No Lidar data received")
            obstacles = []
        else:
            # 障害物の位置を計算
            point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            # 前処理：ノイズを除去し、安全距離以内のポイントのみを考慮
            point_cloud = point_cloud[np.linalg.norm(point_cloud, axis=1) < SAFE_DISTANCE]
            obstacles = [np.array([p[0], p[1], p[2]]) for p in point_cloud]

        # ポテンシャルフィールドを計算
        force = compute_potential_field(current_pos, orientation, obstacles)

        # ドローンの新しい目標位置を計算
        new_pos = current_pos + force * 0.1  # ステップサイズ0.1

        # ドローンを新しい位置に移動
        print(f"Moving to position: {new_pos}")
        client.moveToPositionAsync(new_pos[0], new_pos[1], new_pos[2], 5).join()

        # ゴールに到達したかチェック
        if np.linalg.norm(GOAL_POS - current_pos) < 1.0:
            print("Goal reached!")
            break

        # 短い休止を挟む
        time.sleep(0.1)

finally:
    # API制御を無効化
    client.armDisarm(False)
    client.enableApiControl(False)
