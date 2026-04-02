import torch
import time

def main():
    print("=== GPU VALIDATION ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        # Simple benchmark
        print("\n=== PERFORMANCE MEASUREMENT (Simulated Load) ===")
        device = torch.device("cuda:0")
        
        # Matrix multiplication benchmark
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"100x (1000x1000) Matmul on GPU: {(end - start)*1000:.2f} ms")
    else:
        print("ERROR: CUDA is not available. Please check drivers or installation.")

if __name__ == "__main__":
    main()
