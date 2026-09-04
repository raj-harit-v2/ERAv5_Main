# FP format comparison (0.1 at index 0)

Merged view: scalar/block formats use a **32-element** tensor (index 0 = 0.1, index 1 = 5.0, rest = 0.1). **MXFP4 + MBS** and **MXFP4 + MBS + OAS** use the **128-element** tensor from the MBS/OAS rescue demo. fp32 and Standard MXFP4 already appear above.

| Format Name | Bit Layout | Quantized Value | Abs Error % | Simulation Case |
| :--- | :--- | ---: | ---: | :--- |
| fp32 | `1+8+23` | 0.1 | 0.0000 | Normal |
| fp16 | `1+5+10` | 0.099975586 | 0.0244 | Normal |
| bf16 | `1+8+7` | 0.099609375 | 0.3906 | Normal |
| fp8 E4M3 | `1+4+3` | 0.1015625 | 1.5625 | Normal |
| fp8 E5M2 | `1+5+2` | 0.09375 | 6.2500 | Normal |
| fp4 E2M1 | `1+2+1` | 0 | 100.0000 | Underflow to zero |
| Custom | `1+6+9` | 0.099975586 | 0.0244 | Normal |
| int8 | `8b + shared scale max/127` | 0.11811024 | 18.1102 | Clamped/Scaled |
| MXFP4 | `32-el E2M1 + E8M0` | 0 | 100.0000 | Underflow to zero |
| NVFP4 | `16-el E2M1 + E4M3` | 0 | 100.0000 | Underflow to zero |
| MXFP4 + MBS | `128-el MBS + 32-el E2M1/E8M0` | 0.096191406 | 3.8086 | Rescued! |
| MXFP4 + MBS + OAS | `MBS + 16-el OAS` | 0.096191406 | 3.8086 | Rescued! |
