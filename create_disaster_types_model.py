import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from open_clip import create_model_and_transforms

# 配置参数
config = {
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 20,
    "num_classes": 7,  # 根据你的灾害类别数量修改
    "model_name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "save_path": "./model/disaster_types",
    "train_dir": "./datasets/medic/train",
    "val_dir": "./datasets/medic/val"
}

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess_train, preprocess_val = create_model_and_transforms(
    config["model_name"],
    pretrained=config["pretrained"]
)

# 添加非线性分类头
class NonlinearHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

# 替换分类头
in_features = model.text_projection.shape[1]  # 获取特征维度
model.head = NonlinearHead(in_features, config["num_classes"])
model = model.to(device)

# 数据加载
train_dataset = datasets.ImageFolder(
    config["train_dir"],
    transform=preprocess_train
)
val_dataset = datasets.ImageFolder(
    config["val_dir"],
    transform=preprocess_val
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config["batch_size"]
)

# 优化器和损失函数
optimizer = optim.AdamW([
    {"params": model.visual.parameters(), "lr": config["lr"]/10},
    {"params": model.head.parameters(), "lr": config["lr"]}
])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])

# 训练循环
best_val_acc = 0.0
for epoch in range(config["epochs"]):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        features = model.encode_image(images)
        outputs = model.head(features)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            features = model.encode_image(images)
            outputs = model.head(features)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{config['epochs']}], Val Acc: {val_acc:.2f}%")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "visual": model.visual.state_dict(),
            "head": model.head.state_dict()
        }, f"{config['save_path']}/nonlinear_finetuned.pt")
        torch.save(model.head.state_dict(), f"{config['save_path']}/head.pt")

print("Training completed!")