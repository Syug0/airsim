import airsim
import numpy as np
import time

# AirSimに接続し、先導機を初期化する
client = airsim.MultirotorClient(ip="172.23.32.1", port=41451)
client.confirmConnection()

# 先導機のAPIコントロールを有効にする
client.enableApiControl(True, "leader")
client.armDisarm(True, "leader")
client.takeoffAsync(vehicle_name="leader").join()

# 目標位置をリストとして定義する
target_positions = [
    np.array([18.0, 0.0, -10.0]),  # 前に18ユニット進む
    np.array([18.0, -30.0, -10.0]),  # 左に30ユニット進む
    np.array([48.0, -30.0, -10.0])  # 前に30ユニット進む
]

def move_leader(target_pos):
    """先導機を目標位置に向かって移動させる"""
    while True:
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
        target_position[2] += 0.5  # z座標を少しずつ増加させる
        client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 3, vehicle_name="leader").join()
        time.sleep(1)  # 1秒待機

def main():
    """先導機を順次目標位置に移動させるシナリオを実行する"""
    for target_pos in target_positions:
        move_leader(target_pos)

if __name__ == "__main__":
    main()