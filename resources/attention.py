import torch


class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q_weights = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.k_weights = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.v_weights = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))

    def forward(self, q, k, v):
        q = torch.matmul(q, self.q_weights)
        k = torch.matmul(k, self.k_weights)
        v = torch.matmul(v, self.v_weights)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        return output


model = AttentionModel(input_dim=64, hidden_dim=64, output_dim=10).eval()
q = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
k = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
v = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
inputs = (q, k, v)

output_tensor = model(*inputs)

onnx_program = torch.onnx.export(
    model, inputs, opset_version=23, dynamo=True, report=True
)
