import torch
import matplotlib.pyplot as plt
import sys



if __name__ == "__main__":

    total_samples = 9248 
    global_batch_size = 64*4
    steps_for_one_epoch = int(total_samples / global_batch_size)
    warmup_steps = int(0.25*steps_for_one_epoch)
    epochs = 3
    print(steps_for_one_epoch)
    print(warmup_steps)
    base_lr = 1e-4
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

    def warmup(current_step, warmup_steps, base_lr):
        if current_step <= warmup_steps:
            return float(current_step * base_lr / warmup_steps) / base_lr
        return base_lr


    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: warmup(x, warmup_steps, base_lr))
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_for_one_epoch*epochs-warmup_steps, eta_min=0.1*base_lr, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    lrs = []
    for i in range(int(total_samples/global_batch_size)*epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    plt.plot(lrs)
    plt.show()