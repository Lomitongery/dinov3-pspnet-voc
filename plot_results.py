import matplotlib.pyplot as plt
import numpy as np

# 这是你刚刚跑出来的真实数据！
classes = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse',
           'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']

iou_scores = [97.14, 94.27, 66.98, 94.84, 90.09, 91.11,
              92.51, 92.73, 96.60, 67.75, 93.13, 84.17, 95.90, 92.64,
              91.03, 93.04, 78.91, 93.10, 79.49, 95.22, 78.46]

# 创建一个宽 14，高 7 的高清大图
plt.figure(figsize=(14, 7))

# 绘制柱状图，颜色使用清爽的学术蓝
bars = plt.bar(classes, iou_scores, color='#4C72B0', edgecolor='black', linewidth=1.2)

# 特殊处理：把最低的自行车和椅子标成橙色，突出你的分析！
bars[2].set_color('#DD8452') # Bicycle
bars[9].set_color('#DD8452') # Chair
bars[2].set_edgecolor('black')
bars[9].set_edgecolor('black')

# 在每个柱子上方写上具体的分数
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}',
             ha='center', va='bottom', fontsize=9, rotation=45)

# 设置图表的标题和坐标轴标签
plt.title('Per-class Intersection over Union (IoU) on PASCAL VOC 2012\nOverall mIoU: 88.53% | Pixel Accuracy: 97.54%',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('IoU Score (%)', fontsize=12, fontweight='bold')

# 美化 X 轴标签（旋转 45 度防止重叠）
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(np.arange(0, 105, 10), fontsize=10)

# 添加横向网格线，看起来更专业
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整边距并保存为高清图片
plt.tight_layout()
plt.savefig('iou_bar_chart.png', dpi=300)
print("✅ 图表绘制成功！请在左侧目录查看 iou_bar_chart.png")