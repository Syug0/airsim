import airsim
import numpy as np
import time

# AirSimクライアントの初期化
client = airsim.MultirotorClient("172.23.32.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ドローンの離陸（高度を5mに設定）
target_altitude = -5  # 目標高度を5mに設定
client.takeoffAsync().join()
client.moveToZAsync(target_altitude, 2).join()  # 高度を5mに設定

# LiDARデータを取得して障害物を回避する関数
def avoid_obstacles():
    while True:
        # LiDARセンサデータの取得
        lidar_data = client.getLidarData()
        
        if len(lidar_data.point_cloud) < 3:
            print("No points received from LiDAR")
            continue

        # ポイントクラウドデータの変換
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        # 障害物検出
        obstacle_detected = np.any(points[:, 0] < 5.0)  # 5m以内に障害物があるか確認
        
        if obstacle_detected:
            print("Obstacle detected!")
            # 回避行動（例：左に移動）
            client.moveByVelocityAsync(0, -2, -0.059999, 1).join()
        else:
            # 前進（高度を維持）
            client.moveByVelocityAsync(2, 0, -0.059999, 1).join()

        # 目標高度を維持するための調整
        state = client.getMultirotorState()
        current_altitude = state.kinematics_estimated.position.z_val
        print(f"current altitude {current_altitude}")
        
        # if abs(current_altitude - target_altitude) > 0.5:  # 高度の誤差が0.5m以上の場合に調整
        #     print(f"Adjusting altitude from {current_altitude} to {target_altitude}")
        #     client.moveToZAsync(target_altitude, 1).join()

        time.sleep(0.1)

try:
    avoid_obstacles()
except KeyboardInterrupt:
    client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)
    print("Script ended.")
