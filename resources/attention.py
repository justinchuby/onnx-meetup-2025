import torch


class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, q, k, v):
        # x shape: (seq_len, batch_size, input_dim)
        attn_output, _ = self.attention(q, k, v, need_weights=False)
        # attn_output shape: (seq_len, batch_size, input_dim)
        output = self.output(attn_output)
        # output shape: (seq_len, batch_size, output_dim)
        return output


model = AttentionModel(input_dim=64, hidden_dim=64, output_dim=10).eval()
q = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
k = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
v = torch.randn(10, 2, 64)  # (seq_len, batch_size, input_dim)
inputs = (q, k, v)

output_tensor = model(q, k, v)

onnx_program = torch.onnx.export(
    model, inputs, opset_version=23, dynamo=True, report=True
)
onnx_program.save("attention_model.onnx")
