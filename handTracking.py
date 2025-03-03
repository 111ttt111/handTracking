import cv2  # OpenCV库，用于图像处理和摄像头数据采集
import mediapipe as mp  # MediaPipe库，用于手部检测
import time  # 时间库，用于计算帧率

# 初始化摄像头捕获对象，根据实际情况设置摄像头索引（此处仍使用1，如果只有一个摄像头可修改为0）
cap = cv2.VideoCapture(0)

# 初始化MediaPipe手部检测模块，并设置最大检测手数及检测置信度
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)

# 初始化绘图工具，用于可视化手部关键点和连接线
mpDraw = mp.solutions.drawing_utils
# 设置关键点绘制样式：红色圆圈，粗细5像素
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
# 设置连接线绘制样式：绿色线条，粗细10像素
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

# 用于计算帧率的时间变量
pTime = 0  # 上一帧时间
cTime = 0  # 当前帧时间

# 定义手指关键点索引
FINGER_TIPS = [8, 12, 16, 20]  # 食指、中指、无名指、小指指尖
FINGER_PIPS = [6, 10, 14, 18]  # 食指、中指、无名指、小指PIP关节
THUMB_TIP = 4  # 拇指指尖
THUMB_MCP = 2  # 拇指MCP关节

# 判断非拇指手指是否伸直，通过比较指尖和PIP关节的y坐标
def is_finger_extended(handLms, tip, pip):
    tip_y = handLms.landmark[tip].y
    pip_y = handLms.landmark[pip].y
    return tip_y < pip_y  # y值越小越靠上（在图像中即为“伸直”）

# 判断拇指是否伸直，需要根据左右手分别判断
def is_thumb_extended(handLms, handedness):
    thumb_tip_x = handLms.landmark[THUMB_TIP].x
    thumb_mcp_x = handLms.landmark[THUMB_MCP].x
    # 对于右手，当拇指伸直时，指尖的x坐标会小于MCP关节；左手则相反
    if handedness == "Right":
        return thumb_tip_x < thumb_mcp_x
    else:  # "Left"
        return thumb_tip_x > thumb_mcp_x

# 主循环，持续处理视频流
while True:
    ret, img = cap.read()
    if not ret:
        continue

    # 将图像从BGR颜色空间转换为RGB，因为MediaPipe需要RGB格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用MediaPipe处理图像，检测手部关键点
    result = hands.process(imgRGB)

    # 获取图像尺寸，用于后续坐标转换
    imgHeight, imgWidth = img.shape[0], img.shape[1]

    # 如果检测到手部关键点，并且同时返回左右手信息
    if result.multi_hand_landmarks and result.multi_handedness:
        # 使用zip将手部关键点与左右手信息对应起来
        for handLms, handType in zip(result.multi_hand_landmarks, result.multi_handedness):
            # 获取当前手的左右手标签，值为"Left"或"Right"
            handedness_label = handType.classification[0].label
            # 绘制手部关键点和连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                                  handLmsStyle, handConStyle)

            # 点击手势检测：判断拇指与食指距离是否小于30像素
            thumb_tip = handLms.landmark[4]  # 拇指尖
            index_tip = handLms.landmark[8]  # 食指尖
            thumb_x, thumb_y = int(thumb_tip.x * imgWidth), int(thumb_tip.y * imgHeight)
            index_x, index_y = int(index_tip.x * imgWidth), int(index_tip.y * imgHeight)
            distance_click = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
            if distance_click < 30:
                cv2.putText(img, "Click Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 张开手势检测：判断腕部与食指尖距离是否大于100像素
            wrist = handLms.landmark[0]  # 腕部
            wrist_x, wrist_y = int(wrist.x * imgWidth), int(wrist.y * imgHeight)
            distance_open = ((wrist_x - index_x) ** 2 + (wrist_y - index_y) ** 2) ** 0.5
            if distance_open > 100:
                cv2.putText(img, "Open Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 数字手势检测（1到5）：统计伸直手指的数量
            fingers_extended = 0
            # 检查食指、中指、无名指、小指是否伸直
            for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
                if is_finger_extended(handLms, tip, pip):
                    fingers_extended += 1
            # 根据左右手分别判断拇指是否伸直
            if is_thumb_extended(handLms, handedness_label):
                fingers_extended += 1

            # 显示检测到的数字（1到5）
            if 1 <= fingers_extended <= 5:
                cv2.putText(img, f"Number: {fingers_extended}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 可选：在图像上显示当前手的左右手标签
            cv2.putText(img, handedness_label,
                        (int(handLms.landmark[0].x * imgWidth) - 20, int(handLms.landmark[0].y * imgHeight) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 计算并显示帧率（FPS）
    cTime = time.time()  # 获取当前时间
    fps = 1 / (cTime - pTime) if cTime != pTime else 0  # 防止除零错误
    pTime = cTime  # 更新上一帧时间
    cv2.putText(img, f"FPS: {int(fps)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # 显示处理后的图像
    cv2.imshow('Hand Tracking', img)

    # 检测键盘输入，按q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
