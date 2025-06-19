import gradio as gr
from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载模型
model = YOLO("/workspace/proj_shi/runs/detect/train2/weights/best.pt")
print("Model loaded with classes:", model.names)

def detect_emotion(image):
    print(f"Processing image: {image}")
    
    # 预测并保存结果
    results = model.predict(
        image,
        save=True,
        conf=0.1,    # 进一步降低置信度阈值
        iou=0.3,     # 进一步降低 IOU 阈值
        verbose=True,
        agnostic_nms=True,  # 使用类别无关的 NMS
        max_det=50   # 增加最大检测数量
    )
    
    # 获取预测结果图片路径
    result_path = results[0].save_dir + "/" + os.path.basename(results[0].path)
    print(f"Result saved to: {result_path}")
    
    # 获取检测到的表情类别和数量
    detections = results[0].boxes.data
    print(f"Number of detections: {len(detections)}")
    
    emotions = {}
    for det in detections:
        cls = int(det[5])
        conf = float(det[4])
        label = model.names[cls]
        print(f"Detected: {label} with confidence {conf:.2f}")
        emotions[label] = emotions.get(label, 0) + 1
    
    # 生成检测结果描述
    result_text = "检测结果：\n"
    if emotions:
        for emotion, count in emotions.items():
            result_text += f"{emotion}: {count}个\n"
    else:
        result_text += "未检测到表情"
    
    print(f"Final result text: {result_text}")
    return result_path, result_text

def process_video(frame):
    # 检查输入帧是否有效
    if frame is None:
        return None

    # Gradio 输入的帧是 RGB 格式，OpenCV 和 YOLO plot 需要 BGR 格式
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 预测。通过imgsz参数将输入图像调整为较小尺寸（例如640）
    # 这会极大地加快模型推理速度，从而解决卡顿问题。
    # 模型将在缩小的图像上进行检测，但结果会绘制在原始高分辨率图像上。
    results = model.predict(
        frame_bgr,
        imgsz=640,  # 关键改动：调整图像大小以加速处理
        conf=0.1,
        iou=0.3,
        verbose=False,
        agnostic_nms=True,
        max_det=50
    )
    
    # 在图像上绘制检测结果 (plot返回的是BGR格式)
    result_frame_bgr = results[0].plot()
    
    # 获取检测到的表情类别和数量
    detections = results[0].boxes.data
    emotions = {}
    for det in detections:
        cls = int(det[5])
        label = model.names[cls]
        emotions[label] = emotions.get(label, 0) + 1
    
    # 在图像上添加文本信息
    text = "检测结果: "
    if emotions:
        text += ", ".join([f"{k}: {v}个" for k, v in emotions.items()])
    else:
        text += "未检测到表情"
    
    # 添加文本到 BGR 图像
    cv2.putText(
        result_frame_bgr,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # 将最终的 BGR 图像转换回 RGB 以便 Gradio 显示
    result_frame_rgb = cv2.cvtColor(result_frame_bgr, cv2.COLOR_BGR2RGB)
    
    return result_frame_rgb

# 创建两个界面：图片检测和视频流检测
with gr.Blocks(title="表情识别系统") as demo:
    gr.Markdown("# 表情识别系统")
    gr.Markdown("支持图片检测和实时视频流检测")
    
    with gr.Tab("图片检测"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上传图片")
                image_button = gr.Button("开始检测")
            with gr.Column():
                image_output = gr.Image(type="filepath", label="检测结果")
                text_output = gr.Textbox(label="检测详情")
        
        image_button.click(
            fn=detect_emotion,
            inputs=image_input,
            outputs=[image_output, text_output]
        )
    
    with gr.Tab("实时视频检测"):
        with gr.Row():
            video_input = gr.Image(sources=["webcam"], streaming=True, label="摄像头输入")
            video_output = gr.Image(label="检测结果")
            
        video_input.stream(
            fn=process_video,
            inputs=video_input,
            outputs=video_output
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861) 