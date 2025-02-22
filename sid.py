import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

import numpy as np
import rawpy
import glob
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image



class SeeInDark(nn.Module):
    def __init__(self, num_classes=10):
        super(SeeInDark, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt


def pack_raw(raw,white_level,black_level):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - black_level,0)/ (white_level - black_level) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    #print(f'pack_raw,{out.shape = }')
    return out

class SID_Process():
    def __init__(self,device):
        self.device = device
        model = SeeInDark()
        model.load_state_dict(torch.load('./saved_model/sid.pth',map_location=torch.device('cpu')),strict=True)
        self.model = model.to(device)
    
    def run(self,input_img,ratio,out_img=None):
        print(f'{input_img = }')
        raw = rawpy.imread(input_img)
       
        black_level_per_channel=raw.black_level_per_channel[0]
        white_level=raw.white_level
        print(f'{black_level_per_channel = }')
        print(f'{white_level = }')
        im = raw.raw_image_visible.astype(np.float32) 
        print(f'{im.shape = }')
        input_full = np.expand_dims(pack_raw(im,white_level=white_level,black_level=black_level_per_channel),axis=0) * ratio
        
        # 使用raw的直接后处理时，才使用这个
        # im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0) * ratio	

        input_full = np.minimum(input_full,1.0)

        in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(self.device)
        in_img = F.interpolate(in_img, size=(1424, 2128), mode='bilinear', align_corners=False)
        print(f'{in_img.shape = }')
        out_img = self.model(in_img)
        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output,0),1)
        output = output[0,:,:,:]
       
        return output
       

# 图像处理核心类
class NightEnhancer:
    def __init__(self, device):
        self.sid = SID_Process(device=device)

    def enhance_night_image_single(self, image, ratio):
        """增强单张夜景图像"""
        return self.sid.run(input_img=image, ratio=ratio)

    def enhance_night_image_batch(self, input_folder, output_folder, ratio):
        """批量增强夜景图像"""
        allowed_extensions = {".raw", ".dng", ".arw", ".nef"}
        total_files = len(input_folder)
        for idx, cur_file in enumerate(input_folder):
            cur_file_path = cur_file
            if cur_file_path is None:
                yield "当前文件不存在", None, None, f"进度：{idx}/{total_files} 已处理"
                continue

            cur_file_name, ext = os.path.splitext(cur_file_path.name)
            if ext.lower() not in allowed_extensions:
                yield f"文件：{cur_file_path.name}不是raw文件，略过处理。请使用raw文件（例如：.raw, .dng）。", None, None, f"进度：{idx}/{total_files} 已处理"
                continue

            try:
                raw = rawpy.imread(cur_file_path)
                im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0) * 1
                scale_full = scale_full[0, :, :, :]
                processed_image = (scale_full * 255).astype('uint8')
                result = self.enhance_night_image_single(cur_file_path, ratio)
                output_folder_path = os.path.join(output_folder, cur_file_name + "_enhanced.png")
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)
                Image.fromarray(result, 'RGB').save(output_folder_path)
                yield f"文件{cur_file_path.name}处理完成，已保存到{output_folder_path}", processed_image, result, f"进度：{idx}/{total_files} 已处理"
            except Exception as e:
                yield f"处理文件 {cur_file_path.name} 时发生错误,略过处理，错误信息：{e}", None, None, f"进度：{idx}/{total_files} 已处理"
                continue

        yield "处理完成", None, None, f"进度：{total_files}/{total_files} 已处理"


def export_onnx():
    import onnx
    import onnxsim
    from onnx import shape_inference
    model = SeeInDark()
    input_ = torch.randn(1, 4, 1424, 2128)
    onnx_path = "sid.onnx"
    torch.onnx.export(
        model,        # 模型
        input_,                   # 输入
        onnx_path,                # 导出路径
        opset_version=11,         # ONNX Opset版本
        input_names=["input"],    # 输入名称
        output_names=["output"],  # 输出名称
    )
    print(f"Model exported to {onnx_path}")

    # 3. 简化 ONNX 模型
    simplified_onnx_path = "sid_sim.onnx"
    model = onnx.load(onnx_path)
    model_simplified, check = onnxsim.simplify(model)
    if check:
        onnx.save(model_simplified, simplified_onnx_path)
        print(f"Simplified ONNX model saved to {simplified_onnx_path}")
    else:
        print("Simplification failed!")

    # 4. 使用 ONNX Shape Inference 推导 Feature Map 的 Shape
    inferred_model = shape_inference.infer_shapes(onnx.load(simplified_onnx_path))
    onnx.save(inferred_model, "restormer_inferred.onnx")
    print("Shape inference completed and model saved as 'restormer_inferred.onnx'")

    # 5. 打印推导出的中间层 Feature Map 的 Shape
    for node in inferred_model.graph.value_info:
        name = node.name
        shape = [
            dim.dim_value if dim.dim_value != 0 else "dynamic"
            for dim in node.type.tensor_type.shape.dim
        ]
        print(f"Feature Map Name: {name}, Shape: {shape}")


if __name__=='__main__':
    export_onnx()

