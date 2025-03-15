import torch
import open_clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

# 配置参数
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-5
SAMPLE_RATIO = 0.5  # 20%数据采样

# 数据预处理
def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
    ])

# 创建数据集
train_dataset = datasets.ImageFolder(
    "/home/lixinjian/deep-project/cteate-Dataset/datasets/damage_serverity/train",
    transform=get_transforms()
)
val_dataset = datasets.ImageFolder(
    "/home/lixinjian/deep-project/cteate-Dataset/datasets/damage_serverity/val",
    transform=get_transforms()
)

# 分层采样20%训练数据
indices = list(range(len(train_dataset)))
labels = [label for _, label in train_dataset.samples]
train_idx, _ = train_test_split(
    indices,
    train_size=SAMPLE_RATIO,
    stratify=labels,
    random_state=42
)
train_subset = Subset(train_dataset, train_idx)

# 创建数据加载器
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,
    pin_memory=True
)

# 创建增强模型
class CustomCLIP(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        # self.nonlinear_head = torch.nn.Sequential(
        #     torch.nn.Linear(512, 256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.LayerNorm(256),
        #     torch.nn.Linear(256, num_classes)
        # )
        
        # 增强分类头
        self.nonlinear_head = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, num_classes))
        
        # 冻结前90%的Transformer blocks
        total_blocks = 12
        freeze_until = int(total_blocks * 0.9)
        for name, param in self.backbone.named_parameters():
            if f"blocks.{freeze_until}" in name:
                break
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone.encode_image(x)
        return self.nonlinear_head(features)

model = CustomCLIP(num_classes=len(train_dataset.classes)).cuda()

# 优化器和调度器
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,
    eta_min=LR/100
)
criterion = torch.nn.CrossEntropyLoss()

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

best_acc = 0.0
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    scheduler.step()

    # 验证阶段
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.4f}")
    
    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.backbone.state_dict(), "nonlinear_finetuned.pt")
        torch.save(model.nonlinear_head.state_dict(), "head.pt")
        
    if acc >= 0.95:
        print("达到目标准确率，提前终止训练")
        break

print(f"最高验证准确率: {best_acc:.4f}")