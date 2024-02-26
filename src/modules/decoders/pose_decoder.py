# src/modules/decoders/pose_decoder.py

# one layer MLP
class PoseDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoseDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)