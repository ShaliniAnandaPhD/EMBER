import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, GPT2LMHeadModel, T5ForConditionalGeneration
from torch_geometric.nn import GCNConv, GATConv

class DocumentRetrievalModel(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_classes):
        super(DocumentRetrievalModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
        except RuntimeError as e:
            print(f"Error: {str(e)}. Possible solution: Check if the input tensors have the correct shape and type.")
            raise e

class InstructionGenerationModel(nn.Module):
    def __init__(self, gpt2_model_name):
        super(InstructionGenerationModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            return logits
        except RuntimeError as e:
            print(f"Error: {str(e)}. Possible solution: Ensure that the input tensors have the correct shape and type.")
            raise e

class ScoringModel(nn.Module):
    def __init__(self, t5_model_name):
        super(ScoringModel, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            return logits
        except RuntimeError as e:
            print(f"Error: {str(e)}. Possible solution: Verify that the input tensors have the correct shape and type.")
            raise e

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        try:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        except RuntimeError as e:
            print(f"Error: {str(e)}. Possible solution: Check if the input features and edge indices have the correct shape and type.")
            raise e

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        try:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        except RuntimeError as e:
            print(f"Error: {str(e)}. Possible solution: Ensure that the input features and edge indices have the correct shape and type.")
            raise e

# Example usage and hyperparameter configuration
if __name__ == '__main__':
    # Document Retrieval Model
    bert_model_name = 'bert-base-uncased'
    hidden_size = 768
    num_classes = 2
    document_retrieval_model = DocumentRetrievalModel(bert_model_name, hidden_size, num_classes)

    # Instruction Generation Model
    gpt2_model_name = 'gpt2'
    instruction_generation_model = InstructionGenerationModel(gpt2_model_name)

    # Scoring Model
    t5_model_name = 't5-base'
    scoring_model = ScoringModel(t5_model_name)

    # Graph Convolutional Network
    in_channels = 128
    hidden_channels = 256
    out_channels = 64
    gcn_model = GraphConvolutionalNetwork(in_channels, hidden_channels, out_channels)

    # Graph Attention Network
    num_heads = 8
    gat_model = GraphAttentionNetwork(in_channels, hidden_channels, out_channels, num_heads)