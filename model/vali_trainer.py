import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
import numpy as np
import torch.backends.cudnn as cudnn
import random
import os
import bittensor as bt
import math
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class ValiTrainer:
    def __init__(self, epochs=5, batch_size=128, learning_rate=0.025, momentum=0.9, weight_decay=3e-4, cutout_length=16, grad_clip = 5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.grad_clip = grad_clip
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


   

        # Data loading and normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        # Adding Cutout to the transform
        transform_train.transforms.append(Cutout(self.cutout_length))
        transform_train.transforms.append(transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)))

        g = torch.Generator()
        g.manual_seed(0)

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=5,
                                      worker_init_fn=self.worker_init_fn, generator=g,pin_memory=True,shuffle=False)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ])
        
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, num_workers=5,
                                     worker_init_fn=self.worker_init_fn, generator=g,pin_memory=True)
    
    
    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def set_seed(self, seed=0):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)


    def train(self, model):
        self.set_seed(0)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters()) ## added this 
        optimizer = optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        bt.logging.info(f"Initial learning rate is set to {self.learning_rate}")
        for epoch in range(self.epochs):
            scheduler.step()
            bt.logging.info(f"Epoch {epoch}, LR: {scheduler.get_lr()[0]}")
            # model.droprate = 0.0 * epoch / self.epochs
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the first element if the output is a tuple
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    accuracy = 100 * correct / total
                    bt.logging.info(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, accuracy: {accuracy:.2f}%')
                    running_loss = 0.0
                    correct = 0
                    total = 0

            # Test the model after each epoch
            test_accuracy = self.test(model)

            # Check for overfitting
            # if epoch <= 13 and test_accuracy >= 90:
            #     raise Exception(f"Overfit detected: Epoch {epoch + 1}, Test Accuracy {test_accuracy:.2f}%")

            # if epoch <= 3 and test_accuracy >= 80:
            #     raise Exception(f"Overfit detected: Epoch {epoch + 1}, Test Accuracy {test_accuracy:.2f}%")
    
            # if epoch == 49:
            #     train_accuracy = 100 * correct / total  # Calculate final training accuracy
            #     if abs(train_accuracy - test_accuracy) > 5:
            #         raise Exception(f"Significant accuracy difference detected: Epoch {epoch + 1}, "
            #                         f"Train Accuracy {train_accuracy:.2f}%, Test Accuracy {test_accuracy:.2f}%")

            

        return model

    def test(self, model):
        # self.set_seed(0)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the first 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        bt.logging.info(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
        return 100 * correct / total
    
    def reset_model_weights(self, layer, layer_name=''):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            if hasattr(layer, 'children'):
                for name, child in layer.named_children():
                    child_name = f"{layer_name}.{name}" if layer_name else name
                    if isinstance(child, nn.Conv2d):
                        child.reset_parameters()
                    else:
                        self.reset_model_weights(child, child_name)

    def initialize_weights(self,model):
        self.set_seed(0)
            
        # state_dict = model.state_dict()
        # for name, tensor in model.state_dict().items():
        #     if len(tensor.shape) >= 2:  # Ensure the tensor has at least two dimensions
        #         nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        #     elif len(tensor.shape) == 1:  # Handle biases and 1D tensors separately
        #         if 'bias' in name:
        #             nn.init.constant_(tensor, 0)
        #         elif 'weight' in name:
        #             nn.init.constant_(tensor, 1.0)


        for name, param in model.named_parameters():
            if param.dim() >= 2:  # Ensure the parameter has at least two dimensions
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
            elif param.dim() == 1:  # Handle biases separately if they are one-dimensional
                if 'bias' in name:
                    nn.init.constant_(param.data, 0)
                if 'weight' in name:
                    nn.init.constant_(param.data, 1.0) 
                    # nn.init.uniform_(tensor, -0.05, 0.05)
        self.reset_model_weights(model)
