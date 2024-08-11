def test_loop(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    
    for i, batch in enumerate(test_loader):  # i -->index of batch
        ori_img, noi_img = batch
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noi_img)
        
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)

        # update running training loss
        test_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)

    # avg test loss per epoch
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_baseloss = baseloss / len(test_loader)

    return test_epoch_loss, test_epoch_baseloss


def test_loop_fusion(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    for i, batch in enumerate(test_loader):  # i -->index of batch
        ori_img, noi_img, lidar = batch
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        lidar = lidar.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noi_img,lidar)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)

        # update running training loss
        test_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)

    # avg test loss per epoch
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_baseloss = baseloss / len(test_loader)

    return test_epoch_loss, test_epoch_baseloss

def test_loop_fusion_5(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    for i, batch in enumerate(test_loader):  # i -->index of batch
        ori_img, noi_img, lidar = batch
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        lidar = lidar.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs, k1, k2 = model(noi_img,lidar)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)

        # update running training loss
        test_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)

    # avg test loss per epoch
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_baseloss = baseloss / len(test_loader)

    return test_epoch_loss, test_epoch_baseloss