# Session 10 — Hand Bit Encoding (Task 6)

## Part A — 0.1 (repeating binary fraction)

### FP32 — target 0.1
- Sign: `0`
- Exponent: `01111011`
- Mantissa: `10011001100110011001101`
- Full bits: `0 01111011 10011001100110011001101`
- Stored decimal: `0.10000000149011612`
- Truncation error: `0.000001%`

### BF16 — target 0.1
- Sign: `0`
- Exponent: `01111011`
- Mantissa: `1001100`
- Full bits: `0 01111011 1001100`
- Stored decimal: `0.099609375`
- Truncation error: `-0.390625%`

### FP8 E4M3 — target 0.1
- Sign: `0`
- Exponent: `0011`
- Mantissa: `101`
- Full bits: `0 0011 101`
- Stored decimal: `0.1015625`
- Truncation error: `1.562500%`

## Part B — 1.0 (terminating fraction)

### FP32 — target 1.0
- Sign: `0`
- Exponent: `01111111`
- Mantissa: `00000000000000000000000`
- Full bits: `0 01111111 00000000000000000000000`
- Stored decimal: `1.0`
- Truncation error: `0.000000%`

### BF16 — target 1.0
- Sign: `0`
- Exponent: `01111111`
- Mantissa: `0000000`
- Full bits: `0 01111111 0000000`
- Stored decimal: `1.0`
- Truncation error: `0.000000%`

### FP8 E4M3 — target 1.0
- Sign: `0`
- Exponent: `0111`
- Mantissa: `000`
- Full bits: `0 0111 000`
- Stored decimal: `1.0`
- Truncation error: `0.000000%`

## Training format recommendation

Choose **BF16** for general LLM training because:
- Same exponent range as FP32 (gradients survive to ~1e-38 scale).
- No loss scaling required (unlike FP16).
- FP8 E4M3 is viable for production with external scaling (production FP8 recipes, 2026) but shows
  higher truncation on repeating fractions such as 0.1 (+1.56% in this table).

## Part C — Full format comparison table (0.1 in a 32-element tensor)

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
