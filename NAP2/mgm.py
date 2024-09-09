import copy
import torch


def calculate_mgm(xloader, network, criterion, optimizer, mode, grad=False):
    network.train()
    grads = {}
    for i, (inputs, targets) in enumerate(xloader):
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        # forward
        features, logits = network(inputs)
        loss = criterion(logits, targets)
        # backward
        loss.backward()

        index_grad = 0
        index_name = 0
        for name, param in network.named_parameters():
            if param.grad == None:
                continue
            if index_name > 10: break
            if len(param.grad.view(-1).data[0:100]) < 50: continue
            index_grad = name
            index_name += 1
            if name in grads:
                grads[name].append(copy.copy(param.grad.view(-1).data[0:100]))
            else:
                grads[name] = [copy.copy(param.grad.view(-1).data[0:100])]

        if len(grads[index_grad]) == 50:
            conv = 0
            for name in grads:
                for i in range(50):
                    grad1 = torch.tensor([grads[name][k][i] for k in range(25)])
                    grad2 = torch.tensor([grads[name][k][i] for k in range(25, 50)])
                    grad1 = grad1 - grad1.mean()
                    grad2 = grad2 - grad2.mean()
                    conv += torch.dot(grad1, grad2) / 2500
            break

    return conv
