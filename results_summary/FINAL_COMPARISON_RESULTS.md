# Full Comparison Results: ImageBind vs PCME vs Poly

## 🎯 Test Configuration

### **ImageBind (Raw Embeddings)**
```
Mode: Direct use of ImageBind embeddings
No training required
```

### **PCME (Monte Carlo Baseline)**
```
Loss: pcme_mc (Monte Carlo sampling)
MC Samples: 5 (training), 15 (inference)
Epochs: 40
Learning Rate: 1e-5
Var Reg: kl (weight=0.001)
Batch Size: 64
```

### **Poly (Polynomial Approximation, CIM-Friendly)**
```
Polynomial Degrees: 3, 4, 5, 6
Epochs: 10
Learning Rate: 5e-6  ⭐ (Optimized best value)
Var Reg: kl (weight=0.001)
Batch Size: 64
```

---

## 📊 Full Performance Comparison (MSRVTT-1K Test Set)

| Model | Degree | T2V R@1 | V2T R@1 | T2V R@5 | V2T R@5 | T2V R@10 | V2T R@10 | T2V MedR | V2T MedR | Latency |
|-------|--------|---------|---------|---------|---------|----------|----------|----------|----------|---------|
| **ImageBind** | N/A | **38.70%** | 30.30% | **63.60%** | 53.30% | 72.80% | 64.10% | 3 | 4 | **0.23ms** |
| **PCME** | N/A | **38.80%** | **37.10%** | 63.30% | **62.00%** | **73.80%** | **72.50%** | **2** | **3** | 3.47ms |
| **Poly** | 3 | 36.80% | 35.10% | 61.70% | 59.20% | 72.00% | 70.80% | 3 | 3 | 50.62ms |
| **Poly** | 4 | 36.90% | 35.20% | 62.60% | 60.20% | 72.00% | 70.60% | 3 | 3 | 50.64ms |
| **Poly** | 5 | 35.90% | 34.40% | 61.40% | 60.40% | 71.90% | 70.00% | 3 | 3 | 50.63ms |
| **Poly** | 6 | **37.70%** | 33.20% | 61.90% | 59.60% | 71.70% | 70.30% | 3 | 3 | 50.64ms |




