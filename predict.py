import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 导入你亲手搭的模型
from model import DINOSegmenter

# ==========================================
# 1. PASCAL VOC 的调色板 (给不同物体涂上专属颜色)
# ==========================================
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

def decode_segmap(image, nc=21):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = VOC_COLORMAP[l][0]
        g[idx] = VOC_COLORMAP[l][1]
        b[idx] = VOC_COLORMAP[l][2]
    return np.stack([r, g, b], axis=2)

def predict(image_path, weight_path="pspnet_dino_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在唤醒 GPU: {device} ...")

    # ==========================================
    # 2. 给机器装上你刚刚炼出来的脑子！
    # ==========================================
    model = DINOSegmenter(num_classes=21).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # 开启考试模式！

    # ==========================================
    # 3. 让 AI 看图并作答
    # ==========================================
    print(f"📸 正在加载图片: {image_path}...")
    raw_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(raw_image).unsqueeze(0).to(device)

    print("🧠 AI 正在思考并涂色...")
    with torch.no_grad():
        output = model(img_tensor)
        
    # 提取 AI 的涂色结果
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    colorized_pred = decode_segmap(pred)

    # ==========================================
    # 4. 把原图和 AI 的杰作放在一起展示！
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_image.resize((512, 512)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(colorized_pred)
    axes[1].set_title('Your AI Prediction')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("result.png")
    print("🎉 涂色完成！快去项目文件夹里看看这张名叫 result.png 的大作吧！")

if __name__ == "__main__":
    predict("test.jpg")
