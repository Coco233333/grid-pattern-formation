# -*- coding: utf-8 -*-
import torch
import numpy as np

import os


class Pruner(object):
    def __init__(self, model, trajectory_generator):
        self.model = model
        self.trajectory_generator = trajectory_generator

        self.loss = []
        self.err = []

    def prun_step(self, inputs, pc_outputs, pos, mask):
        ''' 
        Prun on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''

        prune_mask = np.ones(4096).astype("float32") # prune none
        prune_inds = np.random.choice(mask, size = 100, replace = False)
        prune_mask[prune_inds] = 0

        self.model.zero_grad()
        loss, err = self.model.compute_loss_prune(inputs, pc_outputs, pos, prune_mask)
        # loss, err = self.model.compute_loss(inputs, pc_outputs, pos)
        
        return loss.item(), err.item()

    def prun(self, mask, n_steps=10):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for step_idx in range(n_steps):
            inputs, pc_outputs, pos = next(gen)
            loss, err = self.prun_step(inputs, pc_outputs, pos, mask)
            self.loss.append(loss)
            self.err.append(err)

            # print('Step {}/{}. Err: {}cm'.format(
            #     step_idx, n_steps,
            #     np.round(100 * err, 2)))
            
        return self.err