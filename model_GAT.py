import torch
from torch import nn, optim
from torch.nn import Parameter
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np

from torch_geometric.nn import Sequential, GATConv, TransformerConv, MessagePassing
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn.models import InnerProductDecoder, GAE, VGAE
from torch_geometric.nn import GATConv, GAE
from torch_geometric.utils import softmax


# Define GAT-based encoder for GAE
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=1, dropout=0.6)
        self.conv2 = GATConv(32 * 1, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize GAE model with GAT encoder and move it to the GPU
class GAEModel(GAE):
    def __init__(self, in_channels, out_channels):
        encoder = GATEncoder(in_channels, out_channels)
        super(GAEModel, self).__init__(encoder)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass data through the first GAT layer to get attention scores
        _, (edge_index_selfloop, alpha) = self.encoder.conv1(x, edge_index, return_attention_weights=True)
        # matrix shape: number of edges x number of heads
        return edge_index_selfloop,alpha


#NEW START: phenotype-aware GAT model stack
# This encoder is the model-side entry point for phenotype injection. It wraps
# the dual-head layer so the rest of DeepSAS can treat it like a normal GAE encoder.
class PhenotypeGATEncoder(torch.nn.Module):
    """GAE encoder that exposes one phenotype-aware dual-head GAT layer."""

    def __init__(self, in_channels, out_channels, phenotype_embedding_dim=None,
                 age_boundaries=(40.0, 60.0), dropout=0.6):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even for two concatenated heads.")

        phenotype_embedding_dim = (
            phenotype_embedding_dim
            if phenotype_embedding_dim is not None
            else max(1, out_channels // 2)
        )

        self.conv = DualHeadPhenotypeGAT(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            phenotype_embedding_dim=phenotype_embedding_dim,
            age_boundaries=age_boundaries,
            dropout=dropout,
        )

    def forward(self, x, edge_index, ages, is_gene_mask, disease_status=None):
        return self.conv(x, edge_index, ages, is_gene_mask, disease_status)


# This GAE variant exposes the same interface as the standard model while
# requiring the extra phenotype tensors attached to the graph object.
class PhenotypeGAEModel(GAE):
    """Graph autoencoder backed by the phenotype-aware dual-head GAT encoder."""

    def __init__(self, in_channels, out_channels, phenotype_embedding_dim=None,
                 age_boundaries=(40.0, 60.0), dropout=0.6):
        encoder = PhenotypeGATEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            phenotype_embedding_dim=phenotype_embedding_dim,
            age_boundaries=age_boundaries,
            dropout=dropout,
        )
        super().__init__(encoder)

    def get_attention_scores(self, data):
        _, (edge_index, alpha) = self.encoder.conv(
            data.x,
            data.edge_index,
            data.node_ages,
            data.is_gene_mask,
            getattr(data, 'node_disease_status', None),
            return_attention_weights=True,
        )
        return edge_index, alpha


# PhenotypeEmbedding converts cell metadata into node-aligned learnable tokens
# so the heterogeneous gene+cell graph can stay in one homogeneous PyG object.
class PhenotypeEmbedding(nn.Module):
    """Age and disease-status embedding layer for DeepSAS gene/cell graphs.

    DeepSAS stores the heterogeneous graph as a single homogeneous PyG graph:
    gene nodes first, then cell nodes. Phenotype metadata such as age and
    disease status are cell-level quantities, so gene nodes receive universal
    "null" phenotype tokens. Token 0 is reserved for that null state; real
    cell phenotype tokens are shifted by +1.
    """

    def __init__(self, embedding_dim, age_boundaries=(40.0, 60.0),
                 num_disease_categories=2):
        super().__init__()

        # Register boundaries as a buffer so they follow the module across
        # CPU/GPU moves and are saved in state_dict without receiving gradients.
        boundaries = torch.as_tensor(age_boundaries, dtype=torch.float32)
        if boundaries.ndim != 1:
            raise ValueError("age_boundaries must be a 1D sequence of numbers.")
        if boundaries.numel() > 1 and not torch.all(boundaries[1:] > boundaries[:-1]):
            raise ValueError("age_boundaries must be strictly increasing.")
        self.register_buffer("age_boundaries", boundaries)

        # torch.bucketize with B boundaries returns len(B) + 1 possible bins.
        # We add one extra row because row 0 is reserved for gene/null nodes.
        self.age_embedding = nn.Embedding(num_embeddings=boundaries.numel() + 2,
                                          embedding_dim=embedding_dim)
        self.disease_embedding = nn.Embedding(
            num_embeddings=num_disease_categories + 1,
            embedding_dim=embedding_dim,
        )

    @staticmethod
    def build_node_age_inputs(cell_ages, num_genes, device=None, dtype=torch.float32):
        """Create node-aligned age metadata for the DeepSAS graph layout.

        Args:
            cell_ages: 1D array/tensor of continuous ages for cell nodes only,
                ordered exactly like the cell embeddings passed to build_graph_pyg.
            num_genes: Number of gene nodes at the front of the graph.
            device: Optional destination device.
            dtype: Floating dtype for age values.

        Returns:
            node_ages: Shape [num_genes + num_cells]. Gene entries are zero
                placeholders and are ignored by forward().
            is_gene_mask: Boolean shape [num_genes + num_cells]. True for gene
                nodes, False for cell nodes; this matches DeepSAS graph.y.
        """
        cell_ages = torch.as_tensor(cell_ages, dtype=dtype, device=device).view(-1)
        num_cells = cell_ages.numel()

        node_ages = torch.empty(num_genes + num_cells, dtype=dtype, device=device)
        node_ages[:num_genes] = 0.0
        node_ages[num_genes:] = cell_ages

        is_gene_mask = torch.empty(num_genes + num_cells, dtype=torch.bool, device=device)
        is_gene_mask[:num_genes] = True
        is_gene_mask[num_genes:] = False

        return node_ages, is_gene_mask

    @staticmethod
    def build_node_phenotype_inputs(cell_ages, cell_disease_status, num_genes,
                                    device=None, age_dtype=torch.float32):
        """Create node-aligned age, disease, and gene-mask phenotype metadata.

        Disease status is expected as integer cell-level codes:
        -1 = unknown/null, 0 = healthy/control, 1 = diseased/IPF. Gene nodes
        receive 0 placeholders and are mapped to the learnable universal gene
        phenotype token during embedding.
        """
        node_ages, is_gene_mask = PhenotypeEmbedding.build_node_age_inputs(
            cell_ages,
            num_genes,
            device=device,
            dtype=age_dtype,
        )
        cell_disease_status = torch.as_tensor(
            cell_disease_status,
            dtype=torch.long,
            device=device,
        ).view(-1)

        if cell_disease_status.numel() != node_ages.numel() - num_genes:
            raise ValueError(
                "cell_disease_status must have one value per cell node."
            )

        node_disease_status = torch.empty(
            node_ages.numel(),
            dtype=torch.long,
            device=device,
        )
        node_disease_status[:num_genes] = 0
        node_disease_status[num_genes:] = cell_disease_status

        return node_ages, node_disease_status, is_gene_mask

    def forward_components(self, ages, is_gene_mask, disease_status=None):
        """Embed continuous ages and disease status as separate phenotype tokens.

        Args:
            ages: Float tensor of shape [num_nodes]. Values for gene nodes may
                be placeholders because they are never bucketized.
            is_gene_mask: Boolean tensor of shape [num_nodes]. True marks gene
                nodes and forces their phenotype token to 0.
            disease_status: Optional Long tensor of shape [num_nodes]. Cell
                values are 0 for healthy/control and 1 for diseased/IPF.

        Returns:
            Pair of tensors, each with shape [num_nodes, embedding_dim]:
            age embedding and disease-status embedding.
        """
        ages = ages.to(device=self.age_boundaries.device, dtype=self.age_boundaries.dtype)
        is_gene_mask = is_gene_mask.to(device=ages.device, dtype=torch.bool)

        if ages.ndim != 1 or is_gene_mask.ndim != 1:
            raise ValueError("ages and is_gene_mask must both be 1D tensors.")
        if ages.numel() != is_gene_mask.numel():
            raise ValueError("ages and is_gene_mask must have the same length.")

        # Allocate exactly one node-sized LongTensor. This is cheap relative to
        # edge-sized tensors and avoids touching gene ages, which may be NaN.
        phenotype_bins = torch.zeros_like(is_gene_mask, dtype=torch.long)
        cell_mask = ~is_gene_mask

        # right=False gives bins:
        # (-inf, 40) -> young, [40, 60) -> middle-aged, [60, inf) -> old.
        # Shift by +1 so token 0 remains the universal gene/null token. The
        # assignment is valid even if the cell slice is empty, and avoids a
        # host/device sync from checking torch.any(cell_mask) during training.
        phenotype_bins[cell_mask] = (
            torch.bucketize(ages[cell_mask].contiguous(), self.age_boundaries)
            + 1
        )

        if disease_status is None:
            disease_bins = torch.zeros_like(phenotype_bins)
        else:
            disease_status = disease_status.to(device=ages.device,
                                               dtype=torch.long)
            if disease_status.ndim != 1:
                raise ValueError("disease_status must be a 1D tensor.")
            if disease_status.numel() != ages.numel():
                raise ValueError(
                    "disease_status must have the same length as ages."
                )
            disease_bins = torch.zeros_like(phenotype_bins)
            disease_bins[cell_mask] = disease_status[cell_mask] + 1

        return self.age_embedding(phenotype_bins), self.disease_embedding(disease_bins)

    def forward(self, ages, is_gene_mask, disease_status=None):
        """Embed continuous ages and disease status as a combined phenotype token."""
        age_embedding, disease_embedding = self.forward_components(
            ages,
            is_gene_mask,
            disease_status,
        )
        return age_embedding + disease_embedding


# DualHeadPhenotypeGAT is the core phenotype injection layer: one head stays
# expression-driven, while the other head modulates messages with phenotype tokens.
class DualHeadPhenotypeGAT(MessagePassing):
    """Dual-head phenotype-aware GAT layer for large DeepSAS graphs.

    Head 1 uses only transcriptional/node features. Head 2 uses node features
    modulated by an age-bin phenotype embedding. The phenotype head is a gated
    residual over the transcription projection so phenotype context can adjust
    the signal without replacing the expression-driven representation.
    """

    def __init__(self, in_channels, out_channels, phenotype_embedding_dim,
                 age_boundaries=(40.0, 60.0), negative_slope=0.2,
                 dropout=0.0, bias=True, age_gate_init_bias=-2.75,
                 disease_gate_init_bias=-2.75):
        # "add" matches standard GAT aggregation. node_dim=0 keeps this layer
        # compatible with ordinary [num_nodes, num_features] DeepSAS tensors.
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.age_gate_init_bias = age_gate_init_bias
        self.disease_gate_init_bias = disease_gate_init_bias

        self.phenotype_embedding = PhenotypeEmbedding(
            embedding_dim=phenotype_embedding_dim,
            age_boundaries=age_boundaries,
        )

        # Separate transformations for the two semantic heads. Bias is applied
        # after aggregation so source messages stay lean on the 11M-edge graph.
        self.lin_transcription = Linear(in_channels, out_channels, bias=False)
        self.lin_age_delta = Linear(in_channels + phenotype_embedding_dim,
                                    out_channels, bias=False)
        self.lin_age_gate = Linear(in_channels + phenotype_embedding_dim,
                                   out_channels, bias=True)
        self.lin_disease_delta = Linear(in_channels + phenotype_embedding_dim,
                                        out_channels, bias=False)
        self.lin_disease_gate = Linear(in_channels + phenotype_embedding_dim,
                                       out_channels, bias=True)

        # Standard additive GAT attention: a_src^T Wh_j + a_dst^T Wh_i.
        # Keeping these as [out_channels] vectors avoids any edge-wise
        # concatenation of source and target features.
        self.att_transcription_src = Parameter(torch.empty(out_channels))
        self.att_transcription_dst = Parameter(torch.empty(out_channels))
        self.att_phenotype_src = Parameter(torch.empty(out_channels))
        self.att_phenotype_dst = Parameter(torch.empty(out_channels))

        if bias:
            # One bias per returned channel. The layer returns
            # [head1 || head2], so the bias is length 2 * out_channels.
            self.bias = Parameter(torch.empty(2 * out_channels))
        else:
            self.register_parameter("bias", None)

        # These are set only during optional attention-weight returns. Avoid
        # retaining edge-sized tensors during normal training.
        self._alpha = None
        self._return_attention_weights = False

        self.reset_parameters()
    #NEW END: phenotype-aware GAT model stack

    def reset_parameters(self):
        self.phenotype_embedding.age_embedding.reset_parameters()
        self.phenotype_embedding.disease_embedding.reset_parameters()
        self.lin_transcription.reset_parameters()
        self.lin_age_delta.reset_parameters()
        nn.init.zeros_(self.lin_age_gate.weight)
        nn.init.constant_(self.lin_age_gate.bias,
                          self.age_gate_init_bias)
        self.lin_disease_delta.reset_parameters()
        nn.init.zeros_(self.lin_disease_gate.weight)
        nn.init.constant_(self.lin_disease_gate.bias,
                          self.disease_gate_init_bias)
        nn.init.xavier_uniform_(self.att_transcription_src.view(1, -1))
        nn.init.xavier_uniform_(self.att_transcription_dst.view(1, -1))
        nn.init.xavier_uniform_(self.att_phenotype_src.view(1, -1))
        nn.init.xavier_uniform_(self.att_phenotype_dst.view(1, -1))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, ages, is_gene_mask, disease_status=None,
                return_attention_weights=False):
        """Run the dual-head GAT layer.

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels].
            edge_index: PyG COO graph connectivity, shape [2, num_edges].
            ages: Node-aligned continuous age tensor, shape [num_nodes].
            is_gene_mask: Boolean node mask, shape [num_nodes]. True for genes.
            return_attention_weights: If True, also return
                (edge_index, alpha) where alpha has shape [num_edges, 2].
                Leave False during training to minimize VRAM.

        Returns:
            out or (out, (edge_index, alpha)), where out has shape
            [num_nodes, 2 * out_channels].
        """
        if x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, in_channels].")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected x.size(-1) == {self.in_channels}, got {x.size(-1)}."
            )

        self._return_attention_weights = return_attention_weights

        # Head 1: transcriptional projection and node-level attention logits.
        h_tx = self.lin_transcription(x)
        alpha_tx_src = (h_tx * self.att_transcription_src).sum(dim=-1)
        alpha_tx_dst = (h_tx * self.att_transcription_dst).sum(dim=-1)

        out_tx = self.propagate(edge_index, x=h_tx,
                                alpha_src=alpha_tx_src,
                                alpha_dst=alpha_tx_dst,
                                size=None)
        alpha_tx = self._alpha if return_attention_weights else None

        # Head 2: phenotype-conditioned residual. The only concatenations are
        # node-sized [N, F + P], never edge-sized [E, ...]. Gene nodes use the
        # learnable universal phenotype tokens as structural anchors instead
        # of receiving fake cell-level ages or disease labels.
        age_embedding, disease_embedding = self.phenotype_embedding.forward_components(
            ages,
            is_gene_mask,
            disease_status,
        )
        age_embedding = age_embedding.to(dtype=x.dtype)
        disease_embedding = disease_embedding.to(dtype=x.dtype)

        age_input = torch.cat((x, age_embedding), dim=-1)
        disease_input = torch.cat((x, disease_embedding), dim=-1)
        age_delta = self.lin_age_delta(age_input)
        disease_delta = self.lin_disease_delta(disease_input)
        age_gate = torch.sigmoid(self.lin_age_gate(age_input))
        disease_gate = torch.sigmoid(self.lin_disease_gate(disease_input))
        h_ph = h_tx + age_gate * age_delta + disease_gate * disease_delta
        alpha_ph_src = (h_ph * self.att_phenotype_src).sum(dim=-1)
        alpha_ph_dst = (h_ph * self.att_phenotype_dst).sum(dim=-1)

        out_ph = self.propagate(edge_index, x=h_ph,
                                alpha_src=alpha_ph_src,
                                alpha_dst=alpha_ph_dst,
                                size=None)
        alpha_ph = self._alpha if return_attention_weights else None

        out = torch.cat((out_tx, out_ph), dim=-1)
        if self.bias is not None:
            out = out + self.bias

        self._alpha = None
        self._return_attention_weights = False

        if return_attention_weights:
            return out, (edge_index, torch.stack((alpha_tx, alpha_ph), dim=-1))
        return out

    def message(self, x_j, alpha_src_j, alpha_dst_i, index, ptr, size_i):
        """Compute edge messages without edge-wise feature concatenation.

        PyG supplies x_j and attention logits already indexed by edge source
        and target. This method only creates the unavoidable scalar attention
        vector [num_edges] plus the returned message tensor [num_edges, F].
        """
        alpha = alpha_src_j + alpha_dst_i
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self._return_attention_weights:
            self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)

        # self loop is default，so also include the attention of self-loop
        # delete self loop
        self.conv1 = GATConv(dim, dim, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, add_self_loops=False)
        
        # self.conv1 = TransformerConv(dim, dim, heads=1)
        # self.conv2 = TransformerConv(dim, dim, heads=1)

        self.act = torch.nn.CELU()

    def cat(self, x_gene, x_cell, y):
        result = []
        count_gene = 0
        count_cell = 0

        for i in y:
            if i:
                result.append(x_gene[count_gene].view(1, -1))
                count_gene += 1
            else:
                result.append(x_cell[count_cell].view(1, -1))
                count_cell += 1

        result = torch.cat(result)
        return result

    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return x

    def get_att(self, graph):
        x, edge_index,  y = graph.x, graph.edge_index, graph.y
        print(x.shape,y.shape)
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[torch.bitwise_not(y), :]))
        x = self.cat(x_gene, x_cell, y)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x, att = self.conv2(x, edge_index, return_attention_weights=True)

        return x, att


class SenGAE(GAE):
    def __init__(self):
        super(SenGAE, self).__init__(encoder=Encoder(),
                                     decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        z = self.encode(graph)
        # adj_pred = self.decoder(z)
        return z
