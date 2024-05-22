import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, GPT2LMHeadModel, T5ForConditionalGeneration
from torch_geometric.nn import GCNConv, GATConv

class DocumentRetrievalModel(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_classes):
        super(DocumentRetrievalModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        
        # Linear layer for classification
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        try:
            # Pass input through BERT model
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get the pooled output from BERT
            pooled_output = outputs[1]
            
            # Apply dropout to the pooled output
            pooled_output = self.dropout(pooled_output)
            
            # Pass the pooled output through the classifier
            logits = self.classifier(pooled_output)
            
            return logits
        except RuntimeError as e:
            print(f"Error in DocumentRetrievalModel: {str(e)}.")
            print("Possible solution: Check if the input tensors have the correct shape and type.")
            raise e

class InstructionGenerationModel(nn.Module):
    def __init__(self, gpt2_model_name):
        super(InstructionGenerationModel, self).__init__()
        
        # Load pre-trained GPT-2 model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

    def forward(self, input_ids, attention_mask):
        try:
            # Pass input through GPT-2 model
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get the logits from the model output
            logits = outputs.logits
            
            return logits
        except RuntimeError as e:
            print(f"Error in InstructionGenerationModel: {str(e)}.")
            print("Possible solution: Ensure that the input tensors have the correct shape and type.")
            raise e

class ScoringModel(nn.Module):
    def __init__(self, t5_model_name):
        super(ScoringModel, self).__init__()
        
        # Load pre-trained T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, input_ids, attention_mask):
        try:
            # Pass input through T5 model
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get the logits from the model output
            logits = outputs.logits
            
            return logits
        except RuntimeError as e:
            print(f"Error in ScoringModel: {str(e)}.")
            print("Possible solution: Verify that the input tensors have the correct shape and type.")
            raise e

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork, self).__init__()
        
        # First graph convolutional layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Second graph convolutional layer
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        try:
            # Apply first graph convolutional layer
            x = self.conv1(x, edge_index)
            
            # Apply ReLU activation function
            x = F.relu(x)
            
            # Apply dropout regularization
            x = F.dropout(x, training=self.training)
            
            # Apply second graph convolutional layer
            x = self.conv2(x, edge_index)
            
            return x
        except RuntimeError as e:
            print(f"Error in GraphConvolutionalNetwork: {str(e)}.")
            print("Possible solution: Check if the input features and edge indices have the correct shape and type.")
            raise e

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GraphAttentionNetwork, self).__init__()
        
        # First graph attention layer with multiple heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        
        # Second graph attention layer with single head
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        try:
            # Apply first graph attention layer
            x = self.conv1(x, edge_index)
            
            # Apply ELU activation function
            x = F.elu(x)
            
            # Apply dropout regularization
            x = F.dropout(x, training=self.training)
            
            # Apply second graph attention layer
            x = self.conv2(x, edge_index)
            
            return x
        except RuntimeError as e:
            print(f"Error in GraphAttentionNetwork: {str(e)}.")
            print("Possible solution: Ensure that the input features and edge indices have the correct shape and type.")
            raise e

# Example usage and hyperparameter configuration
if __name__ == '__main__':
    # Document Retrieval Model
    bert_model_name = 'bert-base-uncased'  # Name of the pre-trained BERT model
    hidden_size = 768  # Hidden size of the BERT model
    num_classes = 2  # Number of classes for document retrieval
    document_retrieval_model = DocumentRetrievalModel(bert_model_name, hidden_size, num_classes)

    # Instruction Generation Model
    gpt2_model_name = 'gpt2'  # Name of the pre-trained GPT-2 model
    instruction_generation_model = InstructionGenerationModel(gpt2_model_name)

    # Scoring Model
    t5_model_name = 't5-base'  # Name of the pre-trained T5 model
    scoring_model = ScoringModel(t5_model_name)

    # Graph Convolutional Network
    in_channels = 128  # Number of input channels
    hidden_channels = 256  # Number of hidden channels
    out_channels = 64  # Number of output channels
    gcn_model = GraphConvolutionalNetwork(in_channels, hidden_channels, out_channels)

    # Graph Attention Network
    num_heads = 8  # Number of attention heads
    gat_model = GraphAttentionNetwork(in_channels, hidden_channels, out_channels, num_heads)
