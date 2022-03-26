import torch
import torch.nn as nn
import math

from deep_rl.model import TimeDistributed, Flatten, MaskedRNN

class Unflatten(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(-1, *self.shape)

class BigGoalHouseModel(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)        
            d = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.weight.data, -d, d)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # 172
        self.shared_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs, 32, 7, stride = 4), # 42
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, stride = 2), # 20
            nn.ReLU(True),
        ))

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(64, 64, 4, stride = 2), # 9
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1), # 9
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32, 512),
            nn.ReLU()
        ))

        self.main_output_size = 512
        
        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = 512
        self.rnn = MaskedRNN(nn.LSTM(512 + num_outputs + 1, # Conv outputs + last action, reward
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True))

        self._create_pixel_control_network(num_outputs)
        self._create_rp_network()

        self.apply(self.init_weights)
        self.pc_cell_size = 4

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs
        image = observations[0]
        goal = observations[1]
        image, goal = self.shared_base(image), self.shared_base(goal)
        features = torch.cat((image, goal), 2)
        features = self.conv_base(features)
        features = self.conv_merge(features)
        features = torch.cat((features, last_reward_action,), dim = 2)
        return self.rnn(features, masks, states)

    def _create_pixel_control_network(self, num_outputs):
        self.pc_base = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32 * 9 * 9),
            nn.ReLU()
        ))

        self.pc_action = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2), # 20
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size = 4, stride = 2), # 42
            nn.ReLU()
        ))

        self.pc_value = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size = 4, stride = 2), # 20
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_outputs, kernel_size = 4, stride=2), #42
            nn.ReLU()
        ))


    def _create_rp_network(self):
        self.rp = nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32 * 3, 3)
        )

    def reward_prediction(self, inputs):
        observations, _ = inputs
        image = observations[0]
        goal = observations[1]
        image, goal = self.shared_base(image), self.shared_base(goal)
        features = torch.cat((image, goal), 2)
        features = self.conv_base(features)
        features = self.rp(features)
        return features

    def pixel_control(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        features = self.pc_base(features)
        features = features.view(*(features.size()[:2] + (32, 9, 9)))
        action_features = self.pc_action(features)
        features = self.pc_value(features) + action_features - action_features.mean(2, keepdim=True)
        return features, states

    def value_prediction(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        critic = self.critic(features)
        return critic, states

class AuxiliaryBigGoalHouseModel(BigGoalHouseModel):
    def __init__(self, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)
        self._create_deconv_networks()
        self.deconv_cell_size = self.pc_cell_size

    def _create_deconv_networks(self):
        self.deconv_depth = TimeDistributed(nn.Sequential(
            Unflatten(32, 9, 9),
            nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2), #20
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size = 4, stride = 2), #42
        ))

        self.deconv_mask = TimeDistributed(nn.Sequential(
            Unflatten(32, 9, 9),
            nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size = 4, stride = 2)
        ))

        self.deconv_mask_goal = TimeDistributed(nn.Sequential(
            Unflatten(32, 9, 9),
            nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size = 4, stride = 2)
        ))


        self.deconv_depth.apply(self.init_weights)
        self.deconv_mask_goal.apply(self.init_weights)
        self.deconv_mask.apply(self.init_weights)
    
    def forward_deconv(self, inputs, masks, states):
        observations, _ = inputs
        image = observations[0]
        goal = observations[1]
        image, goal = self.shared_base(image), self.shared_base(goal)
        features = torch.cat((image, goal), 2)
        features = self.conv_base(features)

        # heads
        depth = self.deconv_depth(features)
        mask = self.deconv_mask(features)
        mask_goal = self.deconv_mask_goal(features)
        return (depth, mask, mask_goal), states