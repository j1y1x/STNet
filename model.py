import torch.nn as nn        

class MLP(nn.Module):
    def __init__(self, layers, act_func=nn.Tanh()):
        super().__init__()
        self.layers = layers
        self.act_func = act_func
        self.linear_list = []
        for i in range(len(self.layers)-2):
            linear = nn.Linear(self.layers[i],self.layers[i+1])
            self.weight_init(linear)
            self.linear_list.append(linear)
        self.linear_list = nn.ModuleList(self.linear_list)
        linear = nn.Linear(self.layers[-2],self.layers[-1])
        self.weight_init(linear)
        self.fc = linear

    def forward(self, x):
        for i in range(len(self.linear_list)):
            linear = self.linear_list[i]
            x = self.act_func(linear(x))
        linear = self.fc
        y = linear(x)
        return y


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ResNet(nn.Module):
    def __init__(self, in_num, out_num, block_layers, block_num, act_func=nn.Tanh()):
        super().__init__()

        self.block_layers = block_layers
        self.block_num = block_num

        self.in_linear = nn.Linear(in_num, self.block_layers[0])
        self.out_linear = nn.Linear(self.block_layers[-1], out_num)

        self.act_func = act_func

        self.jump_list = []
        self.mlps = []
        for _ in range(self.block_num):
            jump_linear = nn.Linear(self.block_layers[0], self.block_layers[1])
            self.weight_init(jump_linear)
            self.jump_list.append(jump_linear)
            mlp = MLP(block_layers, self.act_func)
            self.mlps.append(mlp)
        self.jump_list = nn.ModuleList(self.jump_list)
        self.mlps = nn.ModuleList(self.mlps)


    def forward(self, x):
        x = self.in_linear(x)
        for i in range(self.block_num):
            mlp = self.mlps[i]
            jump_linear = self.jump_list[i]
            x = mlp(x) + jump_linear(x)
            x = self.act_func(x)
        y = self.out_linear(x)
        return y


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)