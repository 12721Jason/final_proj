import gradio as gr
from ultralytics import YOLO
import os

# 加载模型
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best_smoke.pt")
model = YOLO(model_path)

def detect_smoking(image):
    # 预测并保存结果
    results = model.predict(image, save=True)
    # 获取预测结果图片路径
    result_path = results[0].save_dir + "/" + os.path.basename(results[0].path)
    return result_path

demo = gr.Interface(
    fn=detect_smoking,
    inputs=gr.Image(type="filepath", label="上传图片"),
    outputs=gr.Image(type="filepath", label="检测结果"),
    title="抽烟检测系统",
    description="上传一张图片，自动检测是否有抽烟相关目标"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
