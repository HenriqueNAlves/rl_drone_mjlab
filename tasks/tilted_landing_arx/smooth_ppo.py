"""PPO with an action-smoothness loss term.

Nothing in the objective penalises a *change* in the commanded action between
two control steps: the reward sees the resulting state only, and the rate servo
delivers a command in a few milliseconds, so a twitchy command buys reward
immediately and pays almost nothing back. This adds a quadratic penalty on the
difference between the policy's mean output at consecutive steps::

    loss = surrogate + c_v * value - c_e * entropy
           + coef * mean(||mu(s_t) - mu(s_{t-1})||^2)

Design notes
------------
* The penalty is on the mean action ``mu``, not the sampled one. ``mu`` is what
  the deployed policy produces, and it keeps the term free of exploration
  noise -- sampling would penalise the noise, which is not what should be
  smoothed away.
* Pairs are drawn from the rollout buffer instead of being carried in the
  mini-batches: the mini-batch shuffle destroys adjacency, and two consecutive
  transitions of one environment sit ``num_envs`` apart in the buffer flattened
  as ``[num_steps, num_envs]``. Drawing pairs independently also decouples how
  many pairs are used from the mini-batch size.
* A pair whose predecessor ended an episode is dropped; there is no continuity
  across a reset.
* ``coef`` is a module constant, not a field of ``RslRlPpoAlgorithmCfg``: that
  dataclass is shared by every task, and an unknown kwarg would break the stock
  ``PPO``. Override per run with the ``ARX_SMOOTH_COEF`` environment variable.
* ``update`` is a copy of rsl_rl's ``PPO.update`` with the term inserted. It
  has to be a copy: the penalty must share the backward pass and the gradient
  clipping of the PPO loss to be weighed against it. Re-diff against upstream
  when rsl_rl is upgraded.
"""

from __future__ import annotations

import os

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from torch import nn

ACTION_SMOOTH_COEF = float(os.environ.get("ARX_SMOOTH_COEF", "1.0"))
"""Weight of the smoothness penalty.

Calibration: on the first update of a random policy the penalty is ~0.07 and
its gradient is 1.65x the PPO surrogate gradient at coefficient 1.0. Both
shrink together as the policy smooths out (the penalty fell to ~0.01 within
forty short iterations), so 1.0 stays a strong-but-not-dominant term rather
than a fixed fraction of the loss. Watch ``Loss/action_smooth``; use 0.3 if
learning looks stalled and 3.0 if the policy is still twitchy."""

ACTION_SMOOTH_PAIRS = int(os.environ.get("ARX_SMOOTH_PAIRS", "512"))
"""Number of consecutive-step pairs scored per mini-batch."""


class SmoothPPO(PPO):
    """PPO with a penalty on consecutive-step changes of the mean action."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        action_smooth_coef: float = ACTION_SMOOTH_COEF,
        action_smooth_pairs: int = ACTION_SMOOTH_PAIRS,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.action_smooth_coef = float(action_smooth_coef)
        self.action_smooth_pairs = int(action_smooth_pairs)
        if self.action_smooth_coef > 0.0 and actor.is_recurrent:
            raise ValueError(
                "SmoothPPO samples isolated (s_t, s_{t-1}) pairs, which a recurrent "
                "policy cannot score without replaying the sequence."
            )

    def _action_smoothness_loss(self) -> torch.Tensor:
        """Mean squared change of the policy mean over consecutive control steps."""
        observations = self.storage.observations
        num_steps, num_envs = observations.batch_size[:2]
        zero = torch.zeros((), device=self.device)
        if num_steps < 2:
            return zero

        # Transition (t, n) is `num_envs` ahead of its predecessor (t - 1, n).
        flat_obs = observations.flatten(0, 1)
        flat_dones = self.storage.dones.flatten(0, 1).squeeze(-1).bool()

        current = torch.randint(
            num_envs,
            num_steps * num_envs,
            (self.action_smooth_pairs,),
            device=self.device,
        )
        previous = current - num_envs
        keep = ~flat_dones[previous]
        if not torch.any(keep):
            return zero
        current, previous = current[keep], previous[keep]

        mean_previous = self.actor(flat_obs[previous])
        mean_current = self.actor(flat_obs[current])
        return (mean_current - mean_previous).pow(2).sum(dim=-1).mean()

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_smooth_loss = 0 if self.action_smooth_coef > 0.0 else None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type,
                enabled=self.use_mixed_precision,
                dtype=torch.bfloat16,
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[1],
                )
                # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
                distribution_params = tuple(
                    p[:original_batch_size]
                    for p in self.actor.output_distribution_params
                )
                entropy = self.actor.output_entropy[:original_batch_size]

                # Compute KL divergence and adapt the learning rate
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(
                            batch.old_distribution_params, distribution_params
                        )  # type: ignore
                        kl_mean = torch.mean(kl)

                        # Reduce the KL divergence across all GPUs
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(
                                kl_mean, op=torch.distributed.ReduceOp.SUM
                            )
                            kl_mean /= self.gpu_world_size

                        # Update the learning rate only on the main process
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(
                                self.learning_rate, device=self.device
                            )
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
                )  # type: ignore
                surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
                surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Action smoothness: penalise a change in the mean action between two
                # consecutive control steps.
                smooth_loss = None
                if self.action_smooth_coef > 0.0:
                    smooth_loss = self._action_smoothness_loss()
                    loss = loss + self.action_smooth_coef * smooth_loss

                # RND loss
                rnd_loss = (
                    self.rnd.compute_loss(batch.observations[:original_batch_size])
                    if self.rnd
                    else None
                )  # type: ignore

                # Symmetry loss
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(
                        self.actor, batch, original_batch_size
                    )
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_smooth_loss is not None:
                mean_smooth_loss += smooth_loss.item()  # type: ignore[union-attr]
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Update the normalizers
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_smooth_loss is not None:
            mean_smooth_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if mean_smooth_loss is not None:
            loss_dict["action_smooth"] = mean_smooth_loss
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict
