import torch


class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, q, k, v):
        # x shape: (batch_size, seq_len, input_dim)
        attn_output, _ = self.attention(q, k, v, need_weights=False)
        # attn_output shape: (batch_size, seq_len, input_dim)
        return attn_output


model = AttentionModel(input_dim=64, num_heads=4).eval()
q = torch.randn(2, 10, 64)  # (batch_size, seq_len, input_dim)
k = torch.randn(2, 10, 64)  # (batch_size, seq_len, input_dim)
v = torch.randn(2, 10, 64)  # (batch_size, seq_len, input_dim)
inputs = (q, k, v)

output_tensor = model(q, k, v)

onnx_program = torch.onnx.export(
    model, inputs, opset_version=23, dynamo=True, report=True
)
onnx_program.save("attention_model_23.onnx")

import onnxscript.rewriter.ort_fusions

(ort_model, counts) = onnxscript.rewriter.ort_fusions.optimize_for_ort(onnx_program.model, debug=True)
print(ort_model)
print(counts)
onnx_program.save("attention_model_optimized_23.onnx")