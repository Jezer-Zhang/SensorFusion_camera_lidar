import numpy as np
from matplotlib import pyplot as plt


def plot_losses(histrain, histest, train_baseloss, test_baseloss, zoomin=0):
    histrain = histrain[zoomin:]
    histest = histest[zoomin:]
    epo = len(histrain)
    index = np.arange(1, epo+1, 1, dtype='uint')
    maxLoss = max(max(histrain), max(histest))
    minLoss = min(min(histrain), min(histest))
    plt.plot(index, histrain, c='blue', linestyle='solid')
    plt.plot(index, histest, c='red', linestyle='dashed')
    plt.axhline(train_baseloss, c='blue', linestyle='solid')
    plt.axhline(test_baseloss, c='red', linestyle='dashed')
    # if best in histest:
    #     plt.scatter(histest.index(best), best, marker='*', c='green')
    #     plt.text(histest.index(best), best, 'best model', c='green', horizontalalignment='center')
    # for i in hislr:
    #     if i - zoomin > 0:
    #         i = i - zoomin
    #         plt.plot([i, i], [minLoss, maxLoss * 0.99], c='black', linestyle='solid', linewidth=1.0)
    #         plt.text(i, maxLoss, i, c='black', horizontalalignment='center')
    plt.legend(['Training Loss', 'Test Loss', 'Training Baseline', 'Test Baseline'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plotTraining(histrain, histest, hislr, best, zoomin, zoomout):
    ori_len = len(histrain)
    histrain = histrain[zoomin:zoomout]
    histest = histest[zoomin:zoomout]
    epo = len(histrain)
    index = np.arange(zoomin+1, zoomout+1, 1, dtype='uint')
    maxLoss = max(max(histrain), max(histest))
    minLoss = min(min(histrain), min(histest))
    plt.plot(index, histrain, c='blue', linestyle='solid')
    plt.plot(index, histest, c='red', linestyle='dashed')
    if best in histest:
        histest_list = list(histest)
        plt.scatter(histest_list.index(best)+zoomin, best, marker='*', c='green')
        plt.text(histest_list.index(best)+zoomin, best, 'best model', c='green', horizontalalignment='center')
    # for i in hislr:
    #     if i-zoomin > 0:
    #         i = i - zoomin
            # plt.plot([i,i],[minLoss,maxLoss*0.99], c='black', linestyle='solid', linewidth=1.0)
            # plt.text(i, maxLoss, i, c='black', horizontalalignment='center')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    


def plot_metrics(hispsnr, zoomin=0):
    epo = len(hispsnr)
    index = np.arange(0, epo, 1, dtype='uint')
    plt.plot(index, hispsnr, c='blue', linestyle='solid')
    plt.legend(['PSNR'])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.show()
    return

def plot_validation_img():
    pass
    return

def plotPred(dataloader, fname, classOut, bboxOut, threshold):
    print(fname)
    frame_n = int(fname[0:6])
    frame_amp = dataloader.dataset.framefa[frame_n]

    allBoxes = getbbox(dataloader.dataset.data_dir + '/' + fname)

    bboxes = []
    for i in allBoxes:
        xmin = i['bbox'][1]
        ymin = 15 - i['bbox'][2]
        xmax = i['bbox'][3]
        ymax = 15 - i['bbox'][0]
        if i['category_id'] in dataloader.dataset.classID.keys():
            bbox = [dataloader.dataset.classID[i['category_id']], xmin / 64, ymin / 16, xmax / 64, ymax / 16]
            bboxes.append(bbox)

    classScore = torch.nn.Softmax(dim=1)(classOut)
    bestObj, bestBox = doNMS(dataloader.dataset.config, classScore.data.float(), bboxOut, threshold)

    # draw the result
    frame_amp = np.array(frame_amp)
    img = frame_amp.reshape(16, 64)

    plt.figure(figsize=(10, 4))
    plt.imshow(img)

    numT = len(bboxes)
    numP = len(bestBox)

    ax = plt.gca()
    for i in range(0, numT):
        trueBox = bboxes[i]
        rect_t = Rectangle((trueBox[1] * 64, trueBox[2] * 16),
                           trueBox[3] * 64 - trueBox[1] * 64, trueBox[4] * 16 - trueBox[2] * 16, fill=False,
                           color='white')
        ax.add_patch(rect_t)
        ax.annotate(trueBox[0], (trueBox[1] * 64, trueBox[2] * 16), color='w', weight='bold',
                    fontsize=20, ha='center', va='center')

    for i in range(0, numP):
        predBox = bestBox[i]
        rect_p = Rectangle((predBox[0].item() * 64, predBox[1].item() * 16),
                           predBox[2].item() * 64 - predBox[0].item() * 64,
                           predBox[3].item() * 16 - predBox[1].item() * 16,
                           fill=False, color='red')
        ax.add_patch(rect_p)
        ax.annotate(bestObj[i], (predBox[0].item() * 64, predBox[1].item() * 16), color='r', weight='bold',
                    fontsize=20, ha='center', va='center')
    plt.colorbar()
    plt.show()