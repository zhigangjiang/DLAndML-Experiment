import torch
from torch.optim import Adam
from skimage.segmentation import slic


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def compute_saliency_maps(x, y, model, device):
    model.eval()
    x = x.to(device)

    # 最關鍵的一行 code
    # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
    # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.to(device))
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
    # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
    # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗，
    # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
    # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


layer_activations = None


def filter_explaination(x, model, cnnid, filterid, device, iteration=100, lr=1):
    # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
    # cnnid, filterid: 想要指定第幾層 cnn 中第幾個 filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義的 function 後才可以繼續 forward 下一層 cnn
    # 因此上面的 hook function 中，我們就會把該層的 output，也就是 activation map 記錄下來，這樣 forward 完整個 model 後我們就不只有 loss
    # 也有某層 cnn 的 activation map
    # 注意：到這行為止，都還沒有發生任何 forward。我們只是先告訴 pytorch 等下真的要 forward 時該多做什麼事
    # 注意：hook_handle 可以先跳過不用懂，等下看到後面就有說明了
    # 实际上就是看卷积后的样子。。。，它作为下一层输入，所以说使下一层激活的map

    # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
    model(x.to(device))
    # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
    global layer_activations
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
    # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor

    # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
    x = x.to(device)
    # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
    x.requires_grad_()
    # 我們要對 input image 算偏微分
    optimizer = Adam([x], lr=lr)
    # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)

        objective = -layer_activations[:, filterid, :, :].sum()
        # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
        # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
        # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization
        # 越激活，loss越大

        objective.backward()
        # 計算 filter activation 對 input image 的偏微分
        optimizer.step()
        # 修改 input image 來最大化 filter activation
    filter_visualizations = x.detach().cpu()
    # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

    hook_handle.remove()
    # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
    # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
    # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

    return filter_activations, filter_visualizations


def predict(input, model, device):
    # input: numpy array, (batches, height, width, channels)
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.to(device))
    return output.detach().cpu().numpy()


def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments=100, compactness=1, sigma=1)
