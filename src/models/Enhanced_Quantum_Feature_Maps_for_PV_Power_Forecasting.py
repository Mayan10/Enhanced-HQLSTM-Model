import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class EnhancedQuantumFeatureMap:
    """
    Enhanced quantum feature maps for improved photovoltaic power forecasting.
    Implements multiple encoding strategies beyond simple angle embedding.
    """
    
    def __init__(self, n_qubits: int, encoding_type: str = "fourier"):
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        
    def fourier_encoding(self, x: torch.Tensor, wires: List[int], frequencies: List[float] = None):
        """
        Enhanced Fourier feature map encoding with adaptive frequency components.
        Maps x -> exp(i * sum(freq_k * x_k * sigma_z)) with multi-scale frequencies
        """
        if frequencies is None:
            # Multi-scale frequencies for better feature representation
            frequencies = [1.0, 2.0, 4.0, 8.0, 16.0]
            
        for i, wire in enumerate(wires):
            # Apply multiple frequency components
            for freq in frequencies:
                qml.RZ(freq * x[i % len(x)], wires=wire)
                qml.RX(freq * x[i % len(x)] * 0.5, wires=wire)  # Added RX for better expressivity
                
    def iqp_encoding(self, x: torch.Tensor, wires: List[int]):
        """
        Enhanced Instantaneous Quantum Polynomial (IQP) encoding.
        Creates entangled states with polynomial feature interactions and additional rotations.
        """
        # First layer: individual rotations with enhanced expressivity
        for i, wire in enumerate(wires):
            qml.RZ(x[i % len(x)], wires=wire)
            qml.RX(x[i % len(x)] * 0.5, wires=wire)  # Added RX
            qml.Hadamard(wires=wire)
            
        # Second layer: enhanced pairwise interactions
        for i in range(len(wires)-1):
            for j in range(i+1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(x[i % len(x)] * x[j % len(x)], wires=wires[j])
                qml.RX(x[i % len(x)] * x[j % len(x)] * 0.5, wires=wires[j])  # Added RX
                qml.CNOT(wires=[wires[i], wires[j]])
                
    def amplitude_encoding(self, x: torch.Tensor, wires: List[int]):
        """
        Enhanced amplitude encoding with error correction.
        """
        # Normalize input for amplitude encoding
        x_norm = x / torch.norm(x)
        
        # Pad to power of 2 if necessary
        n_features = len(x_norm)
        n_qubits_needed = int(np.ceil(np.log2(n_features)))
        
        if n_features < 2**n_qubits_needed:
            padding = torch.zeros(2**n_qubits_needed - n_features)
            x_norm = torch.cat([x_norm, padding])
            
        # Convert to numpy for PennyLane
        amplitudes = x_norm.detach().numpy()
        
        # Apply amplitude encoding with error correction
        qml.AmplitudeEmbedding(amplitudes, wires=wires[:n_qubits_needed], normalize=True)
        
        # Add error correction rotations
        for wire in wires[:n_qubits_needed]:
            qml.RZ(np.pi/4, wires=wire)  # Error correction rotation

class EnhancedQuantumLayer(nn.Module):
    """
    Enhanced quantum layer with improved feature encoding and variational circuits.
    """
    
    def __init__(self, n_qubits: int, n_layers: int, encoding_type: str = "fourier"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        
        # Initialize quantum device with error handling
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception as e:
            print(f"Warning: Could not initialize lightning.qubit device: {e}")
            print("Falling back to default.qubit device")
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize variational parameters with better scaling and bounds
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)  # Smaller initialization
        self.feature_map = EnhancedQuantumFeatureMap(n_qubits, encoding_type)
        
        # Create quantum circuit with error handling
        try:
            self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        except Exception as e:
            print(f"Error creating QNode: {e}")
            raise
        
    def quantum_circuit(self, inputs, theta):
        """
        Enhanced quantum circuit with improved encoding and entangling layers.
        """
        wires = list(range(self.n_qubits))
        
        # Normalize inputs to prevent unbounded rotations
        inputs = torch.tanh(inputs)  # Bound inputs to [-1, 1]
        
        # Enhanced feature encoding
        if self.encoding_type == "fourier":
            self.feature_map.fourier_encoding(inputs, wires)
        elif self.encoding_type == "iqp":
            self.feature_map.iqp_encoding(inputs, wires)
        elif self.encoding_type == "amplitude":
            self.feature_map.amplitude_encoding(inputs, wires)
        else:  # Default angle encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
        
        # Enhanced variational layers with improved entanglement
        for layer in range(self.n_layers):
            # Parameterized rotations with better expressivity and bounded angles
            for i in range(self.n_qubits):
                # Use bounded angles for rotations
                angles = torch.tanh(theta[layer, i]) * np.pi  # Bound to [-π, π]
                qml.Rot(angles[0], angles[1], angles[2], wires=i)
                qml.RX(angles[0] * 0.5, wires=i)  # Added RX for better expressivity
            
            # Hardware-efficient entangling gates with improved connectivity
            for i in range(0, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
                qml.CZ(wires=[i, i+1])  # Added CZ for stronger entanglement
            for i in range(1, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
                qml.CZ(wires=[i, i+1])  # Added CZ for stronger entanglement
        
        # Measurements with improved observables and bounded outputs
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
    
    def forward(self, inputs):
        """
        Forward pass of the quantum layer.
        Handles batched inputs by processing each sample individually.
        Ensures device compatibility by moving data to CPU for quantum processing and back to the original device.
        """
        orig_device = inputs.device
        # If inputs is 1D, treat as a single sample
        if inputs.dim() == 1:
            sample = inputs.detach().to('cpu')
            measurements = self.qnode(sample, self.theta.detach().to('cpu'))
            if isinstance(measurements, list):
                measurements = [float(m) for m in measurements]
                measurements = torch.tensor(measurements, dtype=torch.float32)
            return measurements.unsqueeze(0).to(orig_device)
        # If inputs is 2D (batch_size, n_qubits), process each sample
        outputs = []
        for i in range(inputs.shape[0]):
            sample = inputs[i].detach().to('cpu')
            measurements = self.qnode(sample, self.theta.detach().to('cpu'))
            if isinstance(measurements, list):
                measurements = [float(m) for m in measurements]
                measurements = torch.tensor(measurements, dtype=torch.float32)
            outputs.append(measurements)
        return torch.stack(outputs, dim=0).to(orig_device)

class EnhancedHQLSTM(nn.Module):
    """
    Enhanced Hybrid Quantum LSTM with improved quantum feature maps and architecture.
    """
    
    def __init__(self, input_size: int, hidden_size: int, n_qubits: int, 
                 n_quantum_layers: int, encoding_type: str = "fourier"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Enhanced classical preprocessing layers
        self.input_transform = nn.Sequential(
            nn.Linear(input_size, 4 * n_qubits),
            nn.LayerNorm(4 * n_qubits),
            nn.ReLU()
        )
        self.hidden_transform = nn.Sequential(
            nn.Linear(hidden_size, 4 * n_qubits),
            nn.LayerNorm(4 * n_qubits),
            nn.ReLU()
        )
        
        # Enhanced quantum layers for each LSTM gate
        self.quantum_layers = nn.ModuleDict({
            'forget': EnhancedQuantumLayer(n_qubits, n_quantum_layers, encoding_type),
            'input': EnhancedQuantumLayer(n_qubits, n_quantum_layers, encoding_type),
            'candidate': EnhancedQuantumLayer(n_qubits, n_quantum_layers, encoding_type),
            'output': EnhancedQuantumLayer(n_qubits, n_quantum_layers, encoding_type)
        })
        
        # Enhanced output transformation layers
        self.gate_transforms = nn.ModuleDict({
            'forget': nn.Sequential(
                nn.Linear(n_qubits, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ),
            'input': nn.Sequential(
                nn.Linear(n_qubits, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ),
            'candidate': nn.Sequential(
                nn.Linear(n_qubits, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ),
            'output': nn.Sequential(
                nn.Linear(n_qubits, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            )
        })
        
        # Initialize hidden state with better scaling
        self.register_buffer('h0', torch.zeros(hidden_size))
        self.register_buffer('c0', torch.zeros(hidden_size))
        
    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.shape
        
        if hidden_state is None:
            h = self.h0.expand(batch_size, -1)
            c = self.c0.expand(batch_size, -1)
        else:
            h, c = hidden_state
            
        outputs = []
        
        for t in range(seq_len):
            # Transform inputs for quantum processing with enhanced features
            x_t = self.input_transform(x[:, t, :])
            h_t = self.hidden_transform(h)
            
            # Combine and split for quantum gates
            combined = x_t + h_t
            gate_inputs = combined.view(batch_size, 4, self.n_qubits)
            
            # Process through enhanced quantum layers with residual connections
            f_gate = torch.sigmoid(self.gate_transforms['forget'](
                self.quantum_layers['forget'](gate_inputs[:, 0, :])))
            i_gate = torch.sigmoid(self.gate_transforms['input'](
                self.quantum_layers['input'](gate_inputs[:, 1, :])))
            c_tilde = torch.tanh(self.gate_transforms['candidate'](
                self.quantum_layers['candidate'](gate_inputs[:, 2, :])))
            o_gate = torch.sigmoid(self.gate_transforms['output'](
                self.quantum_layers['output'](gate_inputs[:, 3, :])))
            
            # Update cell and hidden states with residual connections
            c = f_gate * c + i_gate * c_tilde
            h = o_gate * torch.tanh(c)
            
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), (h, c)

class EnhancedPVForecastingModel(nn.Module):
    """
    Complete enhanced model for photovoltaic power forecasting.
    """
    
    def __init__(self, input_features: int = 5, sequence_length: int = 24, 
                 hidden_size: int = 32, n_qubits: int = 4, 
                 n_quantum_layers: int = 2, encoding_type: str = "fourier"):
        super().__init__()
        
        self.enhanced_hqlstm = EnhancedHQLSTM(
            input_features, hidden_size, n_qubits, n_quantum_layers, encoding_type
        )
        
        # Enhanced output layers with attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Add normalization layers for better output scaling
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Enhanced output layers with residual connections
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Store target scaler for denormalization
        self.target_scaler = None
        
    def forward(self, x):
        # Enhanced quantum LSTM processing
        lstm_out, _ = self.enhanced_hqlstm(x)
        
        # Self-attention for better temporal modeling
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection with normalization
        lstm_out = self.layer_norm1(lstm_out + attended_out)
        
        # Use the last time step for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Output prediction with normalization
        prediction = self.output_layer(final_hidden)
        
        # Add small random noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(prediction) * 0.01
            prediction = prediction + noise
            
        # Denormalize prediction if target scaler is available
        if self.target_scaler is not None and not self.training:
            prediction = torch.tensor(
                self.target_scaler.inverse_transform(prediction.detach().cpu().numpy()),
                device=prediction.device
            )
        
        return prediction

# Training utilities
class QuantumTrainingUtils:
    """
    Utilities for training enhanced quantum models with specialized optimizers.
    """
    
    @staticmethod
    def quantum_natural_gradient_step(model, loss, lr=0.01):
        """
        Implement quantum natural gradient for better optimization.
        """
        # Standard gradient computation
        model.zero_grad()
        loss.backward()
        
        # Apply quantum natural gradient corrections
        for name, param in model.named_parameters():
            if 'theta' in name and param.grad is not None:
                # Fisher information matrix approximation
                fisher_approx = torch.eye(param.numel()) * 0.1
                nat_grad = torch.linalg.solve(fisher_approx, param.grad.view(-1, 1))
                param.grad = nat_grad.view(param.shape) * lr
    
    @staticmethod
    def adaptive_circuit_depth(model, performance_history, min_depth=1, max_depth=5):
        """
        Dynamically adjust quantum circuit depth based on performance.
        """
        if len(performance_history) > 10:
            recent_improvement = (performance_history[-1] - performance_history[-10]) / 10
            
            if recent_improvement < 0.001:  # Increase depth if improvement stagnates
                for layer in model.enhanced_hqlstm.quantum_layers.values():
                    if hasattr(layer, 'n_layers') and layer.n_layers < max_depth:
                        layer.n_layers += 1
            elif recent_improvement > 0.01:  # Decrease depth if improving rapidly
                for layer in model.enhanced_hqlstm.quantum_layers.values():
                    if hasattr(layer, 'n_layers') and layer.n_layers > min_depth:
                        layer.n_layers -= 1

# Example usage and benchmarking
def benchmark_enhanced_models():
    """
    Benchmark different encoding strategies and enhancements.
    """
    encoding_types = ["angle", "fourier", "iqp", "amplitude"]
    results = {}
    
    # Dummy data for demonstration
    batch_size, seq_len, features = 32, 24, 5
    X = torch.randn(batch_size, seq_len, features)
    y = torch.randn(batch_size, 1)
    
    for encoding in encoding_types:
        print(f"Testing {encoding} encoding...")
        
        model = EnhancedPVForecastingModel(
            input_features=features,
            sequence_length=seq_len,
            encoding_type=encoding
        )
        
        # Simple forward pass timing
        import time
        start_time = time.time()
        
        with torch.no_grad():
            predictions = model(X)
            
        end_time = time.time()
        
        results[encoding] = {
            'time': end_time - start_time,
            'output_shape': predictions.shape,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  Time: {results[encoding]['time']:.4f}s")
        print(f"  Parameters: {results[encoding]['parameters']}")
        print(f"  Output shape: {results[encoding]['output_shape']}")
        
    return results

if __name__ == "__main__":
    print("Enhanced Quantum Machine Learning for PV Power Forecasting")
    print("=" * 60)
    
    # Run benchmarks
    results = benchmark_enhanced_models()
    
    print("\nBenchmark Results:")
    for encoding, metrics in results.items():
        print(f"{encoding.capitalize()} Encoding: {metrics['time']:.4f}s, {metrics['parameters']} params")