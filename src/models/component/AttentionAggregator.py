import torch
import torch.nn.functional as F

class UserAttentionAggregator(torch.nn.Module):
    def __init__(self, dim=400):
        super(UserAttentionAggregator, self).__init__()
        self.attention_weights = torch.nn.Parameter(torch.randn(1, dim))  # 用于计算注意力权重

    def forward(self, tensor1, tensor2, tensor3):
        """
        e.g.:
        tensor1: (batch_size, 50, 400)
        tensor2: (batch_size, 400)
        tensor3: (batch_size, 400)
        """
        # 扩展 tensor2 和 tensor3 到 [batch_size, 50, 400] 维度，以便与 tensor1 配对
        tensor2_expanded = tensor2.unsqueeze(1).expand(-1, 50, -1)  # [batch_size, 50, 400]
        tensor3_expanded = tensor3.unsqueeze(1).expand(-1, 50, -1)  # [batch_size, 50, 400]

        # 计算注意力权重
        attention_weights_1 = torch.matmul(tensor1, tensor2_expanded.transpose(1, 2))  # [batch_size, 50, 50]
        attention_weights_2 = torch.matmul(tensor1, tensor3_expanded.transpose(1, 2))  # [batch_size, 50, 50]

        # 对注意力权重进行缩放和softmax
        attention_weights_1 = F.softmax(attention_weights_1, dim=-1)  # [batch_size, 50, 50]
        attention_weights_2 = F.softmax(attention_weights_2, dim=-1)  # [batch_size, 50, 50]

        # 加权聚合
        weighted_tensor1 = torch.matmul(attention_weights_1, tensor1)  # [batch_size, 50, 400]
        weighted_tensor2 = torch.matmul(attention_weights_2, tensor1)  # [batch_size, 50, 400]

        # 最终的聚合：将加权后的 tensor1、tensor2 和 tensor3 合并
        final_embedding = weighted_tensor1.sum(dim=1) + weighted_tensor2.sum(dim=1)

        return final_embedding

# 示例
batch_size = 32
dim = 400
num_items = 50

tensor1 = torch.randn(batch_size, num_items, dim)  # [32, 50, 400]
tensor2 = torch.randn(batch_size, dim)  # [32, 400]
tensor3 = torch.randn(batch_size, dim)  # [32, 400]

aggregator = AttentionAggregator(dim)
final_embedding = aggregator(tensor1, tensor2, tensor3)  # [32, 400]