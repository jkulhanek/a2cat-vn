import os
from deep_rl.actor_critic import Unreal as UnrealTrainer, UnrealAgent
from deep_rl.actor_critic.unreal.unreal import without_last_item
from deep_rl.actor_critic.unreal.utils import autocrop_observations
from deep_rl.utils import KeepTensor, detach_all, expand_time_dimension, pytorch_call
from deep_rl.common.pytorch import to_tensor
from torch.nn import functional as F
from configuration import configuration
import torch


def compute_auxiliary_target(observations, cell_size = 4, output_size = None):
    with torch.no_grad():
        observations = autocrop_observations(observations, cell_size, output_size = output_size).contiguous()
        obs_shape = observations.size()
        abs_diff = observations.view(-1, *obs_shape[2:])        
        avg_abs_diff = F.avg_pool2d(abs_diff, cell_size, stride=cell_size)
        return avg_abs_diff.view(*obs_shape[:2] + avg_abs_diff.size()[1:])

def compute_auxiliary_targets(observations, cell_size, output_size):
    observations = observations[0]
    return tuple(map(lambda x: compute_auxiliary_target(x, cell_size, output_size), observations[2:]))

class AuxiliaryTrainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auxiliary_weight = 0.05

    def sample_training_batch(self):
        values, report = super().sample_training_batch()
        aux_batch = self.replay.sample_sequence() if self.auxiliary_weight > 0.0 else None
        values['auxiliary_batch'] = aux_batch
        return values, report

    def compute_auxiliary_loss(self, model, batch, main_device):
        loss, losses = super().compute_auxiliary_loss(model, batch, main_device)
        auxiliary_batch = batch.get('auxiliary_batch')

        # Compute pixel change gradients
        if not auxiliary_batch is None:
            devconv_loss = self._deconv_loss(model, auxiliary_batch, main_device)
            loss += (devconv_loss * self.auxiliary_weight)
            losses['aux_loss'] = devconv_loss.item()

        return loss, losses

    def _deconv_loss(self, model, batch, device):
        observations, _, rewards, _ = batch
        observations = without_last_item(observations)
        masks = torch.ones(rewards.size(), dtype = torch.float32, device = device)
        initial_states = to_tensor(self._initial_states(masks.size()[0]), device)
        predictions, _ = model.forward_deconv(observations, masks, initial_states)
        targets = compute_auxiliary_targets(observations, model.deconv_cell_size, predictions[0].size()[3:])
        loss = 0
        for prediction, target in zip(predictions, targets):
            loss += F.mse_loss(prediction, target)

        return loss


class AuxiliaryAgent(UnrealAgent):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def create_model(self, *args):
        model = self.Model(3, self.actions)
        return model

    def wrap_env(self, env):
        return env

    def _initialize(self):
        checkpoint_dir = configuration.get('models_path')

        path = os.path.join(checkpoint_dir, self.name, 'weights.pth')
        model = self.create_model()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()

        @pytorch_call(torch.device('cpu'))
        def step(observations, states):
            with torch.no_grad():
                observations = expand_time_dimension(observations)
                masks = torch.ones((1, 1), dtype=torch.float32)

                policy_logits, _, states = model(observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                return [action.item()], KeepTensor(detach_all(states))

        self._step = step
        return model

    def act(self, obs):
        if self.states is None:
            self.states = self.model.initial_states(1)
        action, self.states = self._step(obs, self.states)
        return action