import wandb
api = wandb.Api()

# 替换为实际的sweep ID
sweep_id = "Guilhoto"
sweep = api.sweep(f"jaxpi/ns_tori/{sweep_id}")
runs = sorted(sweep.runs,
  key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"Best run {runs[0].name} with {val_acc}% validation accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")