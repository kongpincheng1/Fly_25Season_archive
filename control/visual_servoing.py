# 文件名: visual_servoing_controller.py

import cv2
from ultralytics import YOLO
from enum import Enum
import math
import numpy as np

# 这个类中的状态只关心视觉任务本身
class VisualState(Enum):
    GLOBAL_SEARCH = 1
    CENTERING = 2
    TARGET_LOCKED = 3
    LOST = 4

class VisualServoingController:
    """
    一个独立的视觉伺服控制器类。
    它负责处理图像、运行YOLO模型，并根据其内部状态返回指令。
    """
    def __init__(self, model_path, confidence_threshold=0.5, target_class_name='circle', center_tolerance_px=25):
        # 在 __init__ 中，我们只保存参数，不执行任何耗时操作
        print("视觉控制器：对象已创建，模型待加载。")
        self.model_path = model_path
        self.model = None  # 先将模型设置为空
        self.is_model_loaded = False

        self.CONFIDENCE_THRESHOLD = confidence_threshold
        # ... 其他参数保持不变 ...
        self.visual_state = VisualState.GLOBAL_SEARCH
        self.current_target_label = None
        self.initial_target_map = {}

        self.CENTER_TOLERANCE_PX = center_tolerance_px
        self.TARGET_CLASS_NAME = target_class_name
    
    def load_model(self):
        """
        一个独立的方法，专门用于加载模型。
        这个方法应该在ROS节点进入主循环后调用。
        """
        if self.is_model_loaded:
            print("视觉控制器：模型已经加载过了。")
            return
        
        try:
            print("视觉控制器：正在加载模型...")
            self.model = YOLO(self.model_path)
            # 在这里可以进行一次虚拟推理来预热GPU
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False) 
            self.is_model_loaded = True
            print("视觉控制器：模型加载并预热成功。")
            return True
        except Exception as e:
            print(f"视觉控制器：加载模型失败！错误: {e}")
            self.is_model_loaded = False
            return False
    
    def reset_for_new_mission(self):
        """重置整个视觉任务，回到最初的全局搜索状态。"""
        print("视觉控制器：任务重置，返回全局搜索。")
        self.visual_state = VisualState.GLOBAL_SEARCH
        self.current_target_label = None
        self.initial_target_map = {}

    def set_target(self, target_label):
        """
        从外部（ROS节点）设置要追踪的目标。
        这将使视觉状态切换到CENTERING。
        """
        if not self.initial_target_map:
            print("错误：在设置目标前，必须先完成全局搜索！")
            return
        print(f"视觉控制器：已设置新目标 [{target_label}]，开始对准。")
        self.current_target_label = target_label
        self.visual_state = VisualState.CENTERING

    def _find_active_target(self, detections, image_width):
        """（内部方法）根据策略找到当前要追踪的目标。"""
        if not detections:
            return None
        
        target_label = self.current_target_label
        if target_label == "Middle":
            if len(detections) == 3:
                return sorted(detections, key=lambda d: d['center'][0])[1]
            else:
                return min(detections, key=lambda d: abs(d['center'][0] - image_width / 2))
        elif target_label == "Left":
            return min(detections, key=lambda d: d['center'][0])
        elif target_label == "Right":
            return max(detections, key=lambda d: d['center'][0])
        return None

    def _get_drone_command(self, target_center, image_center):
        """（内部方法）生成中文指令。"""
        tx, ty = target_center
        cx, cy = image_center
        dx = tx - cx
        dy = ty - cy

        if abs(dx) <= self.CENTER_TOLERANCE_PX and abs(dy) <= self.CENTER_TOLERANCE_PX:
            return "位置锁定，准备投放"
        
        command = []
        if dx > self.CENTER_TOLERANCE_PX: command.append("向右平移")
        elif dx < -self.CENTER_TOLERANCE_PX: command.append("向左平移")
        if dy > self.CENTER_TOLERANCE_PX: command.append("向后平移")
        elif dy < -self.CENTER_TOLERANCE_PX: command.append("向前平移")
        return " & ".join(command)

    def process_frame(self, frame):
        """
        处理单帧图像的核心方法。
        返回: (visual_state, command, annotated_frame)
        """
        # 在处理第一帧前，确保模型已加载
        if not self.is_model_loaded:
            print("错误：在处理图像前，模型尚未加载！")
            # 返回一个安全的状态
            return self.visual_state, None, frame
        
        height, width, _ = frame.shape
        image_center = (width // 2, height // 2)
        
        # 1. 检测
        results = self.model(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            if box.conf[0] > self.CONFIDENCE_THRESHOLD and self.model.names[int(box.cls[0])] == self.TARGET_CLASS_NAME:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                detections.append({'center': (cx, cy), 'box': [x1, y1, x2, y2]})

        # 2. 视觉状态机逻辑
        command = None
        if self.visual_state == VisualState.GLOBAL_SEARCH:
            if not self.initial_target_map and len(detections) == 3:
                print("视觉控制器：全局搜索成功，已识别3个目标。")
                detections.sort(key=lambda d: d['center'][0])
                self.initial_target_map = {"Left": detections[0], "Middle": detections[1], "Right": detections[2]}
                # 任务完成，等待外部指令
            
        elif self.visual_state == VisualState.CENTERING:
            if not detections:
                self.visual_state = VisualState.LOST
                command = "丢失目标"
            else:
                active_target = self._find_active_target(detections, width)
                if active_target:
                    command = self._get_drone_command(active_target['center'], image_center)
                    if "位置锁定" in command:
                        self.visual_state = VisualState.TARGET_LOCKED
                else:
                    self.visual_state = VisualState.LOST
                    command = "丢失目标"

        # 3. 可视化 (在原图上绘制)
        for det in detections:
            cv2.rectangle(frame, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), (0, 255, 0), 2)
        
        if self.visual_state == VisualState.CENTERING and detections:
             active_target = self._find_active_target(detections, width)
             if active_target:
                cv2.rectangle(frame, (active_target['box'][0], active_target['box'][1]), (active_target['box'][2], active_target['box'][3]), (0, 255, 255), 3)
                cv2.putText(frame, f"Tracking: {self.current_target_label}", (active_target['box'][0], active_target['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.rectangle(frame, (image_center[0] - self.CENTER_TOLERANCE_PX, image_center[1] - self.CENTER_TOLERANCE_PX), (image_center[0] + self.CENTER_TOLERANCE_PX, image_center[1] + self.CENTER_TOLERANCE_PX), (0, 0, 255), 2)
        cv2.putText(frame, f"Visual State: {self.visual_state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return self.visual_state, command, frame