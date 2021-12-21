from layer import *
# from AGCRN import *
import torch

class magnn(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, dropout=0.3, subgraph_size=20, node_dim=40, conv_channels=32, gnn_channels=32, scale_channels=16, end_channels=128, seq_length=12, in_dim=1, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, single_step=True, dynamic_graph=True):
        super(magnn, self).__init__()

        self.num_nodes = num_nodes
        self.dropout = dropout

        self.dynamic_graph = dynamic_graph
        self.device = device
        self.single_step = single_step
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.scale_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        
        self.seq_length = seq_length
        self.layer_num = layers

        if self.dynamic_graph:
            self.gc = dy_graph_constructor(num_nodes, subgraph_size, node_dim, device, layers=self.layer_num, c_in=conv_channels, alpha=tanhalpha)
        else:
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, self.layer_num, device)

        
        if self.single_step:
            self.kernel_set = [7, 6, 3, 2]
        else:
            self.kernel_set = [3, 2, 2]


        self.scale_id = torch.autograd.Variable(torch.randn(self.layer_num, device=self.device), requires_grad=True)
        # self.scale_id = torch.arange(self.layer_num).to(device)
        self.lin1 = nn.Linear(self.layer_num, self.layer_num) 

        self.idx = torch.arange(self.num_nodes).to(device)
        self.scale_idx = torch.arange(self.num_nodes).to(device)


        self.scale0 = nn.Conv2d(in_channels=in_dim, out_channels=scale_channels, kernel_size=(1, self.seq_length), bias=True)

        self.multi_scale_block = multi_scale_block(in_dim, conv_channels, self.num_nodes, self.seq_length, self.layer_num, self.kernel_set)
        # self.agcrn = nn.ModuleList()
        
        length_set = []
        length_set.append(self.seq_length-self.kernel_set[0]+1)
        for i in range(1, self.layer_num):
            length_set.append( int( (length_set[i-1]-self.kernel_set[i])/2 ) )


        for i in range(self.layer_num):
            """
            RNN based model
            """
            # self.agcrn.append(AGCRN(num_nodes=self.num_nodes, input_dim=conv_channels, hidden_dim=scale_channels, num_layers=1) )
            
            if self.dynamic_graph:
                self.gconv1.append(dy_mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(dy_mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))
            else:
                self.gconv1.append(mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))
            
            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=scale_channels,
                                                    kernel_size=(1, length_set[i])))


        self.gated_fusion = gated_fusion(scale_channels, self.layer_num)
        # self.output = linear(self.layer_num*self.hidden_dim, out_dim)
        self.end_conv_1 = nn.Conv2d(in_channels=scale_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)

    def forward(self, input, idx=None):
        seq_len = input.size(3)

        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        
        scale = self.multi_scale_block(input, self.idx)

        # self.scale_weight = self.lin1(self.scale_id)
        
        self.scale_set = [1, 0.8, 0.6, 0.5]

        if self.dynamic_graph:
            adj_matrix = self.gc(self.idx, self.scale_idx, self.scale_set, scale)
        else:
            adj_matrix = self.gc(self.idx, self.scale_idx, self.scale_set)

        outputs = self.scale0(F.dropout(input, self.dropout, training=self.training))

        out = []
        out.append(outputs)
        
        for i in range(self.layer_num):
            """
            RNN-based model
            """
            # output = self.agcrn[i](scale[i].permute(0, 3, 2, 1), adj_matrix) # B T N D
            # output = output.permute(0, 3, 2, 1)
            
            if self.dynamic_graph:
                output = self.gconv1[i](scale[i], adj_matrix[i])+self.gconv2[i](scale[i], adj_matrix[i].transpose(1,2))
            else:
                output = self.gconv1[i](scale[i], adj_matrix[i])+self.gconv2[i](scale[i], adj_matrix[i].transpose(1,0))
            
            
            scale_specific_output = self.scale_convs[i](output)
            
            out.append(scale_specific_output)

            # concatenate
            # outputs = outputs + scale_specific_output

        # mean-pooling    
        # outputs = torch.mean(torch.stack(out), dim=0)

        out0 = torch.cat(out, dim=1)
        out1 = torch.stack(out, dim = 1)
        
        if self.single_step:
            outputs = self.gated_fusion(out0, out1)
        
        x = F.relu(outputs)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x, adj_matrix
