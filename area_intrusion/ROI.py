import cv2
import numpy as np
import json
import os

class MultiPolygonROISelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.all_rois = []  # 存储多个ROI: [[(x1,y1), (x2,y2), ...], ...]
        self.current_roi = []  # 当前正在绘制的ROI点
        self.drawing = False
        self.roi_colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色  
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255), # 紫色
            (0, 255, 255),  # 黄色
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调：绘制多个多边形ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi.append((x, y))
            cv2.circle(self.clone, (x, y), 5, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 实时显示绘制轨迹
            temp_image = self.image.copy()
            self._draw_existing_rois(temp_image)
            
            if len(self.current_roi) > 0:
                # 绘制当前ROI
                for i, pt in enumerate(self.current_roi):
                    cv2.circle(temp_image, pt, 5, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(temp_image, self.current_roi[i-1], pt, (0, 255, 0), 2)
                
                # 预览闭合
                if len(self.current_roi) > 1:
                    cv2.line(temp_image, self.current_roi[-1], (x, y), (0, 255, 0), 2)
            
            self.clone = temp_image
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键完成当前ROI
            if len(self.current_roi) >= 3:
                self.all_rois.append(self.current_roi.copy())
                self.current_roi = []
                self.drawing = False
                self._refresh_display()
                print(f"已添加ROI {len(self.all_rois)}，包含 {len(self.all_rois[-1])} 个点")
    
    def _draw_existing_rois(self, image):
        """绘制所有已存在的ROI"""
        for i, roi_points in enumerate(self.all_rois):
            color = self.roi_colors[i % len(self.roi_colors)]
            
            # 绘制填充区域（半透明）
            overlay = image.copy()
            cv2.fillPoly(overlay, [np.array(roi_points)], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # 绘制边界
            cv2.polylines(image, [np.array(roi_points)], True, color, 3)
            
            # 标注ROI编号
            if roi_points:
                center_x = sum(p[0] for p in roi_points) // len(roi_points)
                center_y = sum(p[1] for p in roi_points) // len(roi_points)
                cv2.putText(image, f"ROI-{i+1}", (center_x-20, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _refresh_display(self):
        """刷新显示"""
        self.clone = self.image.copy()
        self._draw_existing_rois(self.clone)
    
    def select_multiple_rois(self):
        """交互式选择多个ROI"""
        cv2.namedWindow("多区域ROI选择")
        cv2.setMouseCallback("多区域ROI选择", self.mouse_callback)
        
        print("=" * 50)
        print("多区域ROI选择系统")
        print("=" * 50)
        print("操作说明:")
        print("1. 左键点击添加多边形顶点")
        print("2. 右键点击完成当前ROI")
        print("3. 按 'd' 删除上一个ROI") 
        print("4. 按 'c' 确认所有ROI并保存")
        print("5. 按 'r' 重置所有ROI")
        print("6. 按 'q' 退出")
        print("7. 每个ROI至少需要3个顶点")
        print("=" * 50)
        
        while True:
            cv2.imshow("多区域ROI选择", self.clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and self.all_rois:
                break
            elif key == ord('r'):
                self.all_rois = []
                self.current_roi = []
                self.clone = self.image.copy()
                print("已重置所有ROI")
            elif key == ord('d'):
                if self.all_rois:
                    removed = self.all_rois.pop()
                    self._refresh_display()
                    print(f"已删除ROI {len(self.all_rois)+1}")
                else:
                    print("没有可删除的ROI")
            elif key == ord('q'):
                return None
                
        cv2.destroyAllWindows()
        return self.all_rois

def create_multi_roi_mask(image_shape, rois_list):
    """为多个ROI创建组合蒙版"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for roi_points in rois_list:
        if len(roi_points) >= 3:
            points_array = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
    return mask

def save_rois_config(rois_list, image_shape, filepath="rois_config.json"):
    """保存ROI配置到文件"""
    config = {
        "image_width": image_shape[1],
        "image_height": image_shape[0],
        "rois": rois_list,
        "timestamp": str(np.datetime64('now'))
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"ROI配置已保存到: {filepath}")
    return filepath

def load_rois_config(filepath="rois_config.json"):
    """从文件加载ROI配置"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 将列表转换回元组
        rois = [[tuple(point) for point in roi] for roi in config['rois']]
        return rois, (config['image_height'], config['image_width'])
    return None, None

def main():
    # 1. 选择多个湖水区域
    selector = MultiPolygonROISelector("data/dataset/visible/n286.jpg")
    multiple_rois = selector.select_multiple_rois()
    
    if not multiple_rois:
        print("未选择任何ROI区域")
        return
    
    print(f"成功选择 {len(multiple_rois)} 个湖水区域")
    
    # 2. 保存配置
    save_rois_config(multiple_rois, selector.image.shape)
    
    # # 3. 加载测试图像
    # image = selector.image
    
    # # 4. 多区域检测
    # all_detections = multi_roi_detection(
    #     image, multiple_rois, your_detection_model, 
    #     roi_specific_detection=True  # 精确模式
    # )
    
    # # 5. 可视化结果
    # result_image, counts = visualize_multi_roi_detection(image, multiple_rois, all_detections)
    
    # # 6. 显示和保存
    # cv2.imshow("多湖区落水检测", result_image)
    # cv2.imwrite("multi_roi_detection_result.jpg", result_image)
    
    # print("\n检测统计:")
    # for i, count in enumerate(counts):
    #     print(f"  湖区{i+1}: {count} 个目标")
    # print(f"  总计: {sum(counts)} 个目标")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 