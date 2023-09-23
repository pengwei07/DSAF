##################
# 4x-task
##################
import torch
import torch.nn as nn
import torch.nn.functional as F

# laplacian operator--> \nabal^2
lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

# x: \nabal
lap_2d_x = [[[[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [-1/12, 8/12, 0, -8/12, 1/12],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]]]

# y: \nabal
lap_2d_y = [[[[0, 0, -1/12, 0, 0],
            [0, 0, 8/12, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -8/12, 0, 0],
            [0, 0, 1/12, 0, 0]]]]

# Loss_{reg} = \nabla^2 p + \nabla \cdot (\mathbf{u} \cdot \nabla \mathbf{u}). 
class physics(nn.Module):
    def __init__(self):
        super(physics, self).__init__()
        self.dx = 1
        # \nabla^2 p
        self.W_laplace = nn.Conv2d(1, 1, 5, 1, padding=5, bias=False)
        weight_ori1 = (1/self.dx**2)*torch.tensor(lap_2d_op, dtype=torch.float32)
        self.W_laplace.weight = nn.Parameter(weight_ori1)
        self.W_laplace.weight.requires_grad = False 
        # \nabla_x
        self.W_x = nn.Conv2d(1, 1, 5, 1, padding=5, bias=False)
        weight_ori2 = (1/self.dx)*torch.tensor(lap_2d_x, dtype=torch.float32)
        self.W_x.weight = nn.Parameter(weight_ori2)
        self.W_x.weight.requires_grad = False 
        # \nabla_y
        self.W_y = nn.Conv2d(1, 1, 5, 1, padding=5, bias=False)
        weight_ori3 = (1/self.dx)*torch.tensor(lap_2d_y, dtype=torch.float32)
        self.W_y.weight = nn.Parameter(weight_ori3)
        self.W_y.weight.requires_grad = False 
        
    def forward(self, p, u_4):
        # p: (batch_size, 1, h, w)
        # u_4: (batch_size, 4, h, w)

        # Compute the Laplacian of p
        laplace_p = self.W_laplace(p)

        # Split u_4 into its components
        u_10, v_10, u_100, v_100 = torch.split(u_4, 1, dim=1)

        # 10m components
        # Compute the gradient for each component
        grad_u_10_x = self.W_x(u_10)
        grad_u_10_y = self.W_y(u_10)
        grad_v_10_x = self.W_x(v_10)
        grad_v_10_y = self.W_y(v_10)
        # Compute the product of u with its gradient (u dot nabla u for each component)
        product_u_10 = u_10 * grad_u_10_x + v_10 * grad_u_10_y
        product_v_10 = u_10 * grad_v_10_x + v_10 * grad_v_10_y
        # Compute the divergence of the product
        div_product_10_u = self.W_x(product_u_10) + self.W_y(product_u_10)
        div_product_10_v = self.W_x(product_v_10) + self.W_y(product_v_10)
        # Combine the divergences
        div_product_10 = div_product_10_u + div_product_10_v
        
        # 100m components
        # Compute the gradient for each component
        grad_u_100_x = self.W_x(u_100)
        grad_u_100_y = self.W_y(u_100)
        grad_v_100_x = self.W_x(v_100)
        grad_v_100_y = self.W_y(v_100)
        # Compute the product of u with its gradient (u dot nabla u for each component)
        product_u_100 = u_100 * grad_u_100_x + v_100 * grad_u_100_y
        product_v_100 = u_100 * grad_v_100_x + v_100 * grad_v_100_y
        # Compute the divergence of the product
        div_product_100_u = self.W_x(product_u_100) + self.W_y(product_u_100)
        div_product_100_v = self.W_x(product_v_100) + self.W_y(product_v_100)
        # Combine the divergences
        div_product_100 = div_product_100_u + div_product_100_v
        
        # All
        phy_loss_10 = laplace_p + div_product_10
        phy_loss_100 = laplace_p + div_product_100
        phy_loss = phy_loss_10 + phy_loss_100
        
        return phy_loss

def physics_loss(data_sp, data_speed):
    # data_sp: (batch_size, 1, h, w)
    # data_speed: (batch_size, 4, h, w)
    physics_loss = physics().to(data_sp.device)
    phy_loss = physics_loss(data_sp, data_speed)
    return phy_loss

def train(model1, model2, dataloader, criterion, optimizer1, optimizer2, device, minn, maxx):
    model1.train()
    model2.train()
    total_loss = 0
    for input_lr, input_hr, output_lr, output_hr, geo_lr in dataloader:
        input_lr, output_lr, output_hr, geo_lr = (
            input_lr.to(device),
            output_lr.to(device),
            output_hr.to(device),
            geo_lr.to(device).unsqueeze(1)
        )

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        corrected_LR_prediction1, HR_real1, weights1 = model1(input_lr[:,:4,:,:],geo_lr)
        corrected_LR_prediction2, HR_real2, weights2 = model2(input_lr[:,4:,:,:],geo_lr)
        # softmax operation on the weight parameters so that they sum to one
        weights1 = F.softmax(model1.weights, dim=0) 
        weights2 = F.softmax(model2.weights, dim=0) 

        # loss_1
        LR_real1 = (maxx - minn) * corrected_LR_prediction1 + minn
        output_lr1 = (maxx - minn) * output_lr[:,:4,:,:] + minn
        loss1_1 = torch.mean(torch.stack([weights1[i]*criterion(LR_real1[:,i,:,:], output_lr1[:,i,:,:]) for i in range(4)]))

        LR_real2 = (maxx - minn) * corrected_LR_prediction2 + minn
        output_lr2 = (maxx - minn) * output_lr[:,4:,:,:] + minn
        loss1_2 = torch.mean(torch.stack([weights2[i]*criterion(LR_real2[:,i,:,:], output_lr2[:,i,:,:]) for i in range(4)]))
        
        # loss_2
        HR_real1 = (maxx - minn) * HR_real1 + minn
        output_lr1 = (maxx - minn) * output_hr[:,:4,:,:] + minn
        loss2_1 = torch.mean(torch.stack([weights1[i]*criterion(LR_real1[:,i,:,:], output_lr1[:,i,:,:]) for i in range(4)]))

        HR_real2 = (maxx - minn) * corrected_LR_prediction2 + minn
        output_lr2 = (maxx - minn) * output_hr[:,4:,:,:] + minn
        loss2_2 = torch.mean(torch.stack([weights2[i]*criterion(LR_real2[:,i,:,:], output_lr2[:,i,:,:]) for i in range(4)]))
        
        # loss_3
        l_reg = 0.1
        loss_3 = physics_loss(corrected_LR_prediction1[:,0:1,...], corrected_LR_prediction2)
        
        loss1 = loss1_1 + loss1_2
        loss2 = loss2_1 + loss2_2
        loss = (loss1 + loss2) / 2 + l_reg * loss_3
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        total_loss += loss.item() * input_lr.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model1, model2, dataloader, criterion, device, minn, maxx):
    model1.eval()
    model2.eval()
    total_loss = 0
    with torch.no_grad():
        for input_lr, input_hr, output_lr, output_hr, geo_lr in dataloader:
            input_lr, output_lr, output_hr, geo_lr = (
                input_lr.to(device),
                output_lr.to(device),
                output_hr.to(device),
                geo_lr.to(device).unsqueeze(1)
            )

            # Use the corresponding model for the specific channel
            corrected_LR_prediction1, HR_real1, weights1 = model1(input_lr[:,:4,:,:],geo_lr)
            corrected_LR_prediction2, HR_real2, weights2 = model2(input_lr[:,4:,:,:],geo_lr)

            corrected_LR_prediction = torch.cat((corrected_LR_prediction1, corrected_LR_prediction2), dim=1)
            HR_real = torch.cat((HR_real1, HR_real2), dim=1)
            
            # loss_1
            LR_real1 = (maxx - minn) * corrected_LR_prediction1 + minn
            output_lr1 = (maxx - minn) * output_lr[:,:4,:,:] + minn
            loss1_1 = torch.mean(torch.stack([weights1[i]*criterion(LR_real1[:,i,:,:], output_lr1[:,i,:,:]) for i in range(4)]))

            LR_real2 = (maxx - minn) * corrected_LR_prediction2 + minn
            output_lr2 = (maxx - minn) * output_lr[:,4:,:,:] + minn
            loss1_2 = torch.mean(torch.stack([weights2[i]*criterion(LR_real2[:,i,:,:], output_lr2[:,i,:,:]) for i in range(4)]))

            # loss_2
            HR_real1 = (maxx - minn) * HR_real1 + minn
            output_lr1 = (maxx - minn) * output_hr[:,:4,:,:] + minn
            loss2_1 = torch.mean(torch.stack([weights1[i]*criterion(LR_real1[:,i,:,:], output_lr1[:,i,:,:]) for i in range(4)]))

            HR_real2 = (maxx - minn) * corrected_LR_prediction2 + minn
            output_lr2 = (maxx - minn) * output_hr[:,4:,:,:] + minn
            loss2_2 = torch.mean(torch.stack([weights2[i]*criterion(LR_real2[:,i,:,:], output_lr2[:,i,:,:]) for i in range(4)]))

            loss1 = loss1_1 + loss1_2
            loss2 = loss2_1 + loss2_2
            loss = (loss1 + loss2) / 2
            
            total_loss += loss.item() * input_lr.size(0)
    return total_loss / len(dataloader.dataset)