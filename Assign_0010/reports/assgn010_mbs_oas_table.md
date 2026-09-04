# MXFP4 MBS / OAS rescue (0.1 at index 0)

Tensor: 128 elements, index 0 = 0.1, index 1 = 5.0, rest = 0.1.

| Quantization Method | Reads Back (0.1) | Error % | Status |
| :--- | ---: | ---: | :--- |
| fp32 (Reference) | 0.1 | 0.0000 | Normal |
| Standard MXFP4 | 0 | 100.0000 | Underflow to zero |
| MXFP4 + MBS | 0.096191406 | 3.8086 | Rescued! |
| MXFP4 + MBS + OAS | 0.096191406 | 3.8086 | Rescued! |
