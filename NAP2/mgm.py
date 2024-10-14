import copy
import torch
import torchvision
import torchvision.transforms as transforms
from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net
import matplotlib.pyplot as plt


def calculate_mgm(xloader, network, criterion, optimizer, mode, grad=False):
    network.train()
    grads = {}
    for i, (inputs, targets) in enumerate(xloader):
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        # forward
        # ######################
        # print('INPUT TYPE:', type(inputs))
        # state_dict = network.state_dict()
        # first_layer_name = list(state_dict.keys())[0]
        # first_layer_weights = state_dict[first_layer_name]
        # print(f"Data type of {first_layer_name}: {first_layer_weights.dtype}")
        # print(f"Type of {first_layer_name}: {type(first_layer_weights)}")
        # ###########################
        inputs = inputs.to('cuda')
        inputs = inputs.cuda(non_blocking=True)
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

def get_mgm(arch_index, criterion, device):
    # Get a specific architecture from NAS-Bench-201
    network = api.get_net_config(arch_index, 'cifar10')
    # print('network arch:', network)
    # print('converting arch to NN')
    network = get_cell_based_tiny_net(network)
    # print('network object:', type(network))

    # Move the network to GPU if available
    network = network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    mgm = calculate_mgm(trainloader, network, criterion, optimizer, mode='train')

    return mgm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Load NAS-Bench-201 API
    print('Loading CIFAR10 dataset...')
    api = API('C:/Users/alons/PycharmProjects/KNAS/NAP2/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    print('Loading DONE')

    # Define transforms for CIFAR-10
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # # Run a specific architecture
    # arch_index = 123
    # mgm = get_mgm(arch_index, criterion, device)
    # print(f"MGM: {mgm}")

    # Run for a range of Architectures
    mgms = []
    arch_indices = range(10)
    for arch_index in arch_indices:
        mgm = get_mgm(arch_index, criterion, device)
        mgms.append(float(mgm))
        print(f"MGM For Architecture {arch_index}: {mgm}")
    print('MGMs:')
    print(mgms)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(arch_indices, mgms, marker='o')
    plt.title('MGM vs Architecture Index')
    plt.xlabel('Architecture Index')
    plt.ylabel('Mean Gradient Magnitude (MGM)')
    plt.grid(True)
    plt.savefig('mgm_vs_architecture.png')
    plt.show()

