# KRR

This folder has simple codes to do KRR and deltaKRR (with UMA as base model) and then use the model generated to predict.

# KRR_NEB

deltakrr_neb.py in this folder can be used to do delta KRR using sGDML and UMA on the formamide -> formimidic acid reaction. It performs ncalcs number of calculations by adding random errors according to "error_arrays.npz"

# KRR_geomopt

This folder has some simple codes to run geometry optimization using KRR and delta KRR on H2O

# uma_finetune

submit_gpu.sh or submit_gpu_withrandom_error.sh can be used to do finetuning on uma-s model using the data from uccsd_t_result.xyz. 
