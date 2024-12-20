# -*- coding: utf-8 -*-
import torch

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err
    
    # 从example中设置权重（不过为啥example给的转置的？）
    def set_weights(self, weights):
        self.encoder.weight.data = torch.tensor(weights[0].T)
        self.RNN.weight_ih_l0.data = torch.tensor(weights[1].T)
        self.RNN.weight_hh_l0.data = torch.tensor(weights[2].T)
        self.decoder.weight.data = torch.tensor(weights[3].T)
        # model.encoder.weight.data 对应于 weights[0]，形状为 (512, 4096)。
        # model.RNN.weight_ih_l0.data 对应于 weights[1]，形状为 (2, 4096)。
        # model.RNN.weight_hh_l0.data 对应于 weights[2]，形状为 (4096, 4096)。
        # model.decoder.weight.data 对应于 weights[3]，形状为 (4096, 512)。


    # 修剪部分速度输入
    def prune_forward(self, inputs, mask):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        with torch.no_grad():
            v, p0 = inputs
            g = [self.encoder(p0)[None]]  # initial state

            # Get RNN weights
            W_ih = self.RNN.weight_ih_l0  # input to hidden weights
            W_hh = self.RNN.weight_hh_l0  # hidden to hidden weights
            
            # Manually implement the RNN forward pass
            for i in range(v.shape[0]):  # v.shape[1] is the sequence length
                # Compute the current time step hidden state
                velocity_input = torch.matmul(v[i], W_ih.t()) * mask
                h_t = torch.relu(velocity_input + torch.matmul(g[-1], W_hh.t()))
                g.append(h_t)
        return torch.stack(g[1:], dim=1)[0]
    
    
    def predict_prune(self, inputs, mask):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.prune_forward(inputs, mask))
        
        return place_preds


    def compute_loss_prune(self, inputs, pc_outputs, pos, mask):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict_prune(inputs, mask)
        yhat = self.softmax(self.predict_prune(inputs, mask))
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

    # 修剪read out层的神经元
    def compute_loss_prune_read_out(self, inputs, pc_outputs, pos, mask):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        original_weights = self.decoder.weight.data.clone()

        with torch.no_grad():
            self.decoder.weight.data[:, mask] = 0

        preds = self.predict(inputs)

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()
        
        # 恢复原始权重
        self.decoder.weight.data = original_weights

        return err