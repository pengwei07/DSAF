##################
# 2x-task
##################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from data_load import data_load
from model import DSAF
from train import train, evaluate
##################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data load
train_loader, val_loader, test_loader, maxx, minn = data_load()

# main
def main():
    # Initialize the parameters
    # The initial weight is set according to the variance of each weather factor correction
    aa = 0.87
    bb = 1.61
    cc = 1.89
    dd = 1.51
    ee = 1.17
    ff = 1.26
    gg = 1.66
    hh = 1.77
    init_weights1 = torch.tensor((aa,bb,cc,dd)).float().to(device)
    init_weights2 = torch.tensor((ee,ff,gg,hh)).float().to(device)

    model1 = DSAF(in_channels=4, num_res_blocks=3, num_heads=4, patch_dim=2, growth_rate=16, num_dense_layers=3,
                in_channels1=8, num_res_blocks1=3, num_dense_layers1=3, growth_rate1=16, init_weights=init_weights1).to(device)

    model2 = DSAF(in_channels=4, num_res_blocks=3, num_heads=4, patch_dim=2, growth_rate=16, num_dense_layers=3,
                in_channels1=8, num_res_blocks1=3, num_dense_layers1=3, growth_rate1=16, init_weights=init_weights2).to(device)
    # optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0005, weight_decay=1e-5)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0005, weight_decay=1e-5)
    # scheduler
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.8, patience=15, verbose=True)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.8, patience=15, verbose=True)
    # criterion
    criterion = nn.MSELoss()
    ########################
    # main
    ########################
    num_epochs = 200
    best_val_loss = float('inf')
    loss_train = np.zeros(num_epochs)
    loss_val = np.zeros(num_epochs)
    loss_test = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train(model1, model2, train_loader, criterion, optimizer1, optimizer2, device)
        val_loss = evaluate(model1, model2, val_loader, criterion, device)
        test_loss = evaluate(model1, model2, test_loader, criterion, device)

        loss_train[epoch] = train_loss
        loss_val[epoch] = val_loss
        loss_test[epoch] = test_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model1.state_dict(), '2x_model1.pth')
            torch.save(model2.state_dict(), '2x_model2.pth')
            
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == num_epochs:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Test Loss: {test_loss:.8f}")
            
        # Update learning rate
        scheduler1.step(val_loss)
        scheduler2.step(val_loss)
            
    end_time = time.time()
    training_time = end_time - start_time
    print("Training took {} mins".format(training_time/60))
    
main()