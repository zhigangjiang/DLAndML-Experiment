import torch
import torch.nn.functional as F


class Attacker:
    def __init__(self, model, loader, mean, std, device):
        # 讀入預訓練模型 vgg16
        self.model = model
        self.loader = loader
        self.device = device
        self.mean = mean
        self.std = std

    # FGSM 攻擊
    @staticmethod
    def fgsm_attack(image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad  # fgsm核心思想：梯度上升，直接到边界上，否则传统attack梯度上升太高会超过eps，导致图片不像原图，然后又要进行迭代拉回
        return perturbed_image

    @staticmethod
    def naive_attack(image, epsilon, data_grad, iteration):
        # 找出 gradient 的方向
        perturbed_image = image.copy()
        for i in range(iteration):
            perturbed_image = perturbed_image - 0.01 * data_grad.detach().cpu()
            if torch.norm(perturbed_image - image) < epsilon:
                break
        return perturbed_image

    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for data in self.loader:
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            data_raw = inputs
            inputs.requires_grad = True  # 关键代码，我们目的是对图片进行对loss的梯度上升，即让loss越大
            # 將圖片丟入 model 進行測試 得出相對應的 class
            outputs = self.model(inputs)
            outputs_pred = outputs.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if outputs_pred.item() != labels.item():
                wrong += 1
                continue

            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            inputs_grad = inputs.grad.data
            perturbed_data = self.fgsm_attack(inputs, epsilon, inputs_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class
            outputs = self.model(perturbed_data)
            final_pred = outputs.max(1, keepdim=True)[1]

            if final_pred.item() == labels.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device=self.device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=self.device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                    data_raw = data_raw * torch.tensor(self.std, device=self.device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=self.device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()

                    # nn输出label ， attacked输出label，原始数据，attacked得到数据
                    adv_examples.append((outputs_pred.item(), final_pred.item(), data_raw, adv_ex))
                else:
                    pass
        final_acc = (fail / (wrong + success + fail))

        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc
