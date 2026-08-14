# Assign_007 Evidence

Overall: **PASS**

Problem ID: 4 (Fourier alternative)
Schema: a07.4.2
Star arm: `fourier_canine`
Policy: `fourier_v1|code512|freq32|rope|canine=1|vq=0|untied|tok_37b74fda18ad`

## Gates

| Gate | Result | Detail |
|------|--------|--------|
| problem_id_locked | PASS | problem_id=4 |
| fourier_policy_type | PASS | fourier_codepoint_sum |
| tokenizer_hash_present | PASS | 37b74fda18adfee3 |
| rope_position_policy | PASS | rope |
| collision_audit_written | PASS | collision_report keys |
| stage_audits_complete | PASS | n=4 |
| ablation_ran | PASS | dense,fourier,fourier_canine,kronecker |
| figures_core_present | PASS | ['collision_heatmap_scripts.png', 'fourier_wave_composition.png', 'problem4_proof_metrics.png', 'stage_audit_dashboard.png'] |
| unicode_policy_requires_text | PASS | seam-crossing declared |
| canine_arm_present | PASS | fourier_canine |
| vq_gated_correctly | PASS | ENABLE_VQ=False |
| fourier_family_beats_kronecker_on_one_indic_metric | PASS | star=fourier_canine hi_ce=0.6866 base_hi=0.6917 k=1.0378; disc=1.000/0.833; coll=0/1 |
| english_ce_soft_regression | PASS | star_en=4.4029 k_en=1.0500 tol=1.5 |

## Ablation summary

| Arm | val_ce_all | val_ce_en | val_ce_hi | disc_acc | params |
|-----|-----------|-----------|-----------|----------|--------|
| dense | 1.9095 | 3.1799 | 0.7089 | 1.000 | 165120 |
| kronecker | 1.0437 | 1.0500 | 1.0378 | 0.833 | 263488 |
| fourier | 2.7442 | 4.9160 | 0.6917 | 1.000 | 165184 |
| fourier_canine | 2.4923 | 4.4029 | 0.6866 | 1.000 | 706688 |

## Figures

- `param_memory_budget.png`
- `v5_param_budget_reference.png`
- `kronecker_byte_grid_example.png`
- `fourier_wave_composition.png`
- `pos_dim_truncation_bars.png`
- `collision_heatmap_scripts.png`
- `frozen_shift_grad_spike.png`
- `stage_audit_dashboard.png`
- `problem4_proof_metrics.png`
- `train_val_loss_curves.png`
- `upgrade_ladder_hi_ce.png`
