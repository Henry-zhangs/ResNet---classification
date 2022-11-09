import torchvision.transforms as transforms     # 预处理
import torch
from PIL import Image
from model import ResNet

def predict(img):
    data_transform =transforms.Compose([
                                     transforms.Resize((32,32)),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    net = ResNet()
    net.load_state_dict(torch.load('./resNet34.pth',map_location='cpu'))  # 加载网络训练的参数
    im = Image.open(img)
    im = data_transform(im)  # 图像维度 （C，H，W）
    im = torch.unsqueeze(im,dim  = 0)  # 增加维度，第0维增加1 ，维度（1，C，H，W）

    net.eval()          # 打开eval模式
    with torch.no_grad():
        outputs = net(im)       # 预测图像
        outputs = torch.softmax(outputs,dim = 1)    # 让结果经过softmax

        value,predict = outputs.topk(k = 5,dim = 1) # 取出最大的前五个结果
        value =  (value*100).numpy()

        labels = []             # 前五个分类的类别
        pro = []                # 前五个类别的概率

        for i in zip(predict,value):
            labels.append(classes[int(i[0][0])])
            labels.append(classes[int(i[0][1])])
            labels.append(classes[int(i[0][2])])
            labels.append(classes[int(i[0][3])])
            labels.append(classes[int(i[0][4])])

            pro.append(str(i[1][0])+'%')
            pro.append(str(i[1][1])+'%')
            pro.append(str(i[1][2])+'%')
            pro.append(str(i[1][3])+'%')
            pro.append(str(i[1][4])+'%')

        return labels,pro

