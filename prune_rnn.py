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

        self.model.zero_grad()
        loss, err = self.model.compute_loss_prune(inputs, pc_outputs, pos, mask)
        # loss, err = self.model.compute_loss(inputs, pc_outputs, pos)
        
        return loss.item(), err.item()

    def prun(self, mask, n_steps=10, mask_size=100):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        for i in range(mask_size):
            prune_mask = np.ones(4096).astype("float32") # prune none
            prune_inds = np.random.choice(mask, size = i, replace = False)
            prune_mask[prune_inds] = 0
            errs = 0

            # Construct generator
            gen = self.trajectory_generator.get_generator()

            # 每次修剪都迭代n_steps次
            for j in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                _, err = self.prun_step(inputs, pc_outputs, pos, prune_mask)
                errs+=err

            errs/=n_steps
            self.err.append(errs)
            
        return self.err

    # 修剪read out层的神经元
    def prun_step_read_out(self, inputs, pc_outputs, pos, mask):
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

        self.model.zero_grad()
        err = self.model.compute_loss_prune_read_out(inputs, pc_outputs, pos, mask)
        # loss, err = self.model.compute_loss(inputs, pc_outputs, pos)
        
        return err.item()

    def prun_read_out(self, mask, n_steps=10, mask_size=100, step_by_step = True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''
        if step_by_step == True:
            for i in range(mask_size):
                prune_inds = np.random.choice(mask, size = i, replace = False)
                errs = 0

                # Construct generator
                gen = self.trajectory_generator.get_generator()

                # 每次修剪都迭代n_steps次
                for j in range(n_steps):
                    inputs, pc_outputs, pos = next(gen)
                    err = self.prun_step_read_out(inputs, pc_outputs, pos, prune_inds)
                    errs+=err

                errs/=n_steps
                self.err.append(errs)
        
        else:
            prune_inds = np.random.choice(mask, size = mask_size, replace = False)

            # Construct generator
            gen = self.trajectory_generator.get_generator()

            # 每次修剪都迭代n_steps次
            for j in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                err = self.prun_step_read_out(inputs, pc_outputs, pos, prune_inds)
                self.err.append(err)
            
        return self.err