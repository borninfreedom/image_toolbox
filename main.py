import os
import rawpy
import numpy as np
from PIL import Image
import gradio as gr
import torch
from sid import SID_Process,NightEnhancer

# 检查设备支持
class DeviceChecker:
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device:", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        return device

# Gradio 界面类
class GradioUI:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.demo = self.create_ui()

    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# The Image Toolbox Is All You Need")
            demo.css = """
            #batch-process-button {
                background-color: #007AFF;
                color: white;
            }
            """
            with gr.Tabs() as tabs:
                self.create_night_enhance_tab()
                self.create_exif_parser_tab()
                self.create_image_resize_tab()
        return demo

    def create_night_enhance_tab(self):
        with gr.TabItem("极暗夜景增强", id="night_enhance"):
            mode = gr.Radio(["单个文件处理", "批量处理"], label="模式选择", value="单个文件处理")
            single_file_row, batch_file_row = self.create_single_file_ui(), self.create_batch_file_ui()
            mode.change(self.update_ui_mode, inputs=[mode], outputs=[single_file_row, batch_file_row])

    def create_single_file_ui(self):
        with gr.Row(visible=True) as single_file_row:
            with gr.Column():
                with gr.Row():
                    input_image = gr.File(label="输入RAW图像文件")
                    output_image = gr.Image(label="输出图像")
                with gr.Row():
                    ratio_single_bar = gr.Slider(label="提亮强度", minimum=0, maximum=300, value=100, step=10)
                with gr.Row():
                    process_button_single = gr.Button("处理", elem_id="batch-process-button")
                with gr.Row():
                    error_message_box = gr.Textbox(label="错误提示", value="", visible=False, interactive=False)
                with gr.Row():
                    gr.Markdown("""
                    <span style='font-size: 18px;'></span>  \n
                    <span style='font-size: 18px;'>请传入RAW图</span>  \n
                    <span style='font-size: 18px;'>当前算法只能处理肉眼看起来极暗的照片，如果处理正常曝光的图片，处理后就会过曝。</span>
                    """)
                examples_data = [
                    ["assets/input_0475.png", 300, "assets/out_0475.png"],
                    ["assets/input_0139.png", 1, "assets/out_0139.png"]
                ]
                with gr.Row():
                    gr.Examples(examples=examples_data, inputs=[input_image, ratio_single_bar, output_image], label="示例表格")
                process_button_single.click(
                    self.validate_and_process_single,
                    inputs=[input_image, ratio_single_bar],
                    outputs=[output_image, error_message_box, error_message_box]
                )
        return single_file_row

    def create_batch_file_ui(self):
        with gr.Row(visible=False) as batch_file_row:
            with gr.Column():
                with gr.Row(scale=1):
                    input_folder = gr.File(label="选择输入文件夹，选择到最底层文件夹即可，不要选择单个文件", file_count="directory")
                with gr.Row(scale=1):    
                    output_folder = gr.Dropdown(choices=self.list_non_hidden_files(os.path.expanduser("~/Pictures")), label="选择输出文件夹")
                with gr.Row():    
                    ratio_batch_bar = gr.Slider(label="提亮强度", minimum=1, maximum=300, value=100, step=1)
                with gr.Row():
                    batch_process_button = gr.Button("处理", elem_id="batch-process-button")
                with gr.Row():
                    error_message_box = gr.Textbox(label="提示", value="", visible=True, interactive=False)
                progress_display = gr.Textbox(label="处理进度", interactive=False)
                with gr.Row():
                    input_image_display = gr.Image(label="当前输入图像", interactive=False)
                    output_image_display = gr.Image(label="当前输出图像", interactive=False)
                batch_process_button.click(
                    self.image_processor.enhance_night_image_batch,
                    inputs=[input_folder, output_folder, ratio_batch_bar],
                    outputs=[error_message_box, input_image_display, output_image_display, progress_display],
                    queue=True
                )
        return batch_file_row

    def create_exif_parser_tab(self):
        with gr.TabItem("EXIF解析", id="exif_parser"):
            gr.Markdown("EXIF解析功能开发中...")

    def create_image_resize_tab(self):
        with gr.TabItem("图像resize", id="image_resize"):
            gr.Markdown("图像resize功能开发中...")

    @staticmethod
    def list_non_hidden_files(path):
        """列出指定路径下的非隐藏文件和文件夹"""
        if not os.path.exists(path):
            return []
        return [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

    @staticmethod
    def update_ui_mode(selected_mode):
        if selected_mode == "单个文件处理":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    def validate_and_process_single(self, file, ratio):
        """验证文件格式并处理单张图像"""
        allowed_extensions = {".raw", ".dng", ".arw", ".nef"}
        if file is None:
            return None, "请上传一个文件。", gr.update(visible=True)
        _, ext = os.path.splitext(file.name)
        if ext.lower() not in allowed_extensions:
            return None, f"文件格式错误：{ext}。请上传RAW格式文件（例如：.raw, .dng）。", gr.update(visible=True)
        result = self.image_processor.enhance_night_image_single(file, ratio)
        return result, "", gr.update(visible=False)

    # def validate_and_process_batch(self, input_folder, output_folder, ratio):
    #     """验证文件格式并处理批量图像"""
    #     #return self.image_processor.enhance_night_image_batch(input_folder, output_folder, ratio)
    #     error_message, input_image, output_image, progress_dis = self.image_processor.enhance_night_image_batch(input_folder, output_folder, ratio)
    #     return  error_message, input_image, output_image, progress_dis

    def launch(self):
        self.demo.launch(share=False)

# 主函数
def main():
    device = DeviceChecker.get_device()
    image_processor = NightEnhancer(device)
    gradio_ui = GradioUI(image_processor)
    gradio_ui.launch()

if __name__ == "__main__":
    main()