import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class RelationGNNLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__(aggr='add')
        self.node_transform = nn.Linear(node_dim, out_dim)
        self.edge_transform = nn.Linear(edge_dim, out_dim)
        self.attention = nn.Linear(out_dim * 3, 1)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.node_transform(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        edge_features = self.edge_transform(edge_attr)
        triple_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        
        alpha = F.leaky_relu(self.attention(triple_features))
        alpha = F.softmax(alpha, dim=-1)
        return x_j * alpha + edge_features

class SubpartGNN(nn.Module):
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RelationGNNLayer(
                node_dim if i == 0 else hidden_dim,
                edge_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.output_transform = nn.Linear(hidden_dim, node_dim)
        
    def forward(self, x, edge_index, edge_attr):
        h = x
        for layer in self.layers:
            h_new = layer(h, edge_index, edge_attr)
            h = h + h_new
            h = self.node_norm(h)
            h = F.relu(h)
        return self.output_transform(h)

class GNNEmbeddingEnricher:
    def __init__(self, base_model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        self.gnn = SubpartGNN(
            node_dim=self.bert_dim,
            edge_dim=self.bert_dim,
            hidden_dim=768
        )
        
    def get_subpart_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for a subpart"""
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**tokens)
            return outputs.last_hidden_state[:, 0, :]
            
    def get_relation_embedding(self, relation: str) -> torch.Tensor:
        """Get embedding for a relation"""
        return self.get_subpart_embedding(relation)
        
    def create_graph_data(self, 
                         subparts: List[str],
                         relations: List[str],
                         triples: List[Tuple[int, int, int]]) -> Data:
        """Create PyTorch Geometric Data object"""
        node_features = []
        for subpart in subparts:
            emb = self.get_subpart_embedding(subpart)
            node_features.append(emb)
        node_features = torch.cat(node_features, dim=0)
        
        edge_features = []
        edge_index = []
        
        for s1, s2, r in triples:
            edge_index.extend([[s1, s2], [s2, s1]])
            rel_emb = self.get_relation_embedding(relations[r])
            edge_features.extend([rel_emb, rel_emb])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features = torch.cat(edge_features, dim=0)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
    def enrich_embeddings(self,
                         subparts: List[str],
                         relations: List[str],
                         triples: List[Tuple[int, int, int]]) -> torch.Tensor:
        """Enrich subpart embeddings using GNN"""
        
        graph_data = self.create_graph_data(subparts, relations, triples)
    
        enriched_features = self.gnn(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr
        )
        
        return enriched_features
        
    def enrich_sentence_embedding(self,
                                sentence_embedding: torch.Tensor,
                                subparts: List[str],
                                relations: List[str],
                                triples: List[Tuple[int, int, int]]) -> torch.Tensor:
        """Enrich full sentence embedding using GNN outputs"""

        enriched_subparts = self.enrich_embeddings(subparts, relations, triples)
        

        attention = torch.matmul(sentence_embedding, enriched_subparts.t())
        attention = F.softmax(attention, dim=-1)

        context_vector = torch.matmul(attention, enriched_subparts)
        
        alpha = 0.3  
        enriched_sentence = sentence_embedding + alpha * context_vector
        
        return F.normalize(enriched_sentence, p=2, dim=-1)

def example_usage():
    subparts = [
        "The big brown dog",
        "chased",
        "the small cat",
        "in the park"
    ]
    relations = ["subject", "object", "location"]
    triples = [(1, 0, 0), (1, 2, 1), (1, 3, 2)]
    
    enricher = GNNEmbeddingEnricher()
    
    sentence_embedding = torch.randn(1, enricher.bert_dim)
    
    enriched_embedding = enricher.enrich_sentence_embedding(
        sentence_embedding,
        subparts,
        relations,
        triples
    )
     
    enriched_subparts = enricher.enrich_embeddings(
        subparts,
        relations,
        triples
    )
    
    return enriched_embedding, enriched_subparts

if __name__ == "__main__":
    enriched_embedding, enriched_subparts = example_usage()
    print(enriched_embedding.shape)
    print(enriched_subparts.shape)
    
