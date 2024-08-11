def train_loop(train_loader, model, criterion, optimizer_model, device):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):  # i -->index of batch
        ori_img, noi_img = batch # cpu maybe
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        # transform -- gpu --createdataset 
        

        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noi_img)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)
        # clear the gradients of all optimized variables
        optimizer_model.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_model.step()
        # update running training loss
        train_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)
    # print avg training statistics
    train_epoch_loss = train_loss / len(train_loader)
    train_epoch_baseloss = baseloss / len(train_loader)
    return train_epoch_loss, train_epoch_baseloss


def train_loop_fusion(train_loader, model, criterion, optimizer_model, device):
    model.train()
    train_loss = 0
    
    for i, batch in enumerate(train_loader):  # i -->index of batch
        ori_img, noi_img, lidar = batch # cpu maybe
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        lidar = lidar.to(device)
        # transform -- gpu --createdataset 
        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noi_img, lidar)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)
        # clear the gradients of all optimized variables
        optimizer_model.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_model.step()
        # update running training loss
        train_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)
    # print avg training statistics
    train_epoch_loss = train_loss / len(train_loader)
    train_epoch_baseloss = baseloss / len(train_loader)
    return train_epoch_loss, train_epoch_baseloss

def train_loop_fusion_5(train_loader, model, criterion, optimizer_model, device):
    model.train()
    train_loss = 0
    
    for i, batch in enumerate(train_loader):  # i -->index of batch
        ori_img, noi_img, lidar = batch # cpu maybe
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        lidar = lidar.to(device)
        # transform -- gpu --createdataset 
        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs, k1, k2 = model(noi_img, lidar)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)
        # clear the gradients of all optimized variables
        optimizer_model.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_model.step()
        # update running training loss
        train_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)
    # print avg training statistics
    train_epoch_loss = train_loss / len(train_loader)
    train_epoch_baseloss = baseloss / len(train_loader)
    return train_epoch_loss, train_epoch_baseloss

def train_loop_resnet(train_loader, model, criterion, optimizer_model, device):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):  # i -->index of batch
        image, label = batch # cpu maybe
        ori_img = ori_img.to(device)
        noi_img = noi_img.to(device)
        # transform -- gpu --createdataset 

        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noi_img)
        # calculate the loss
        loss = criterion(outputs, ori_img)
        baseloss = criterion(ori_img, noi_img)
        # clear the gradients of all optimized variables
        optimizer_model.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_model.step()
        # update running training loss
        train_loss += loss.item() * ori_img.size(0)
        baseloss += baseloss.item() * ori_img.size(0)
    # print avg training statistics
    train_epoch_loss = train_loss / len(train_loader)
    train_epoch_baseloss = baseloss / len(train_loader)
    return train_epoch_loss, train_epoch_baseloss