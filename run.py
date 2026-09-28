"""Codes-only outcome prediction on MIMIC-IV with PyHealth 2.0.2.

  .venv/Scripts/python run.py --task mortality   [--dev]
  .venv/Scripts/python run.py --task readmission [--dev]

mortality:   codes of admission i -> death during admission i+1
readmission: codes of admission i -> readmitted within 30 days of discharge
Inputs are ICD diagnoses, ICD procedures, NDC prescriptions of admission i.
"""
import argparse
from datetime import timedelta

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import Transformer
from pyhealth.tasks import MortalityPredictionMIMIC4, ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer

# Windows spawns worker processes by re-importing this file; without the
# guard each worker re-runs the whole script and spawns its own workers.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mortality", "readmission"], required=True)
    p.add_argument("--root", default="data/mimiciv/3.1")
    p.add_argument("--dev", action="store_true", help="first 1000 patients only")
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()

    ds = MIMIC4EHRDataset(
        root=a.root,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir="cache",
        dev=a.dev,
    )
    task = (MortalityPredictionMIMIC4() if a.task == "mortality"
            else ReadmissionPredictionMIMIC4(window=timedelta(days=30)))
    samples = ds.set_task(task)
    print(f"{len(samples)} samples")

    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=0)
    dl = lambda d, s: get_dataloader(d, batch_size=64, shuffle=s)

    # model = Transformer(dataset=samples)
    # trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc", "f1"], output_path="output")
    # trainer.train(dl(train, True), dl(val, False), epochs=a.epochs, monitor="pr_auc")
    
    from models.logistic_regression import LogisticRegression

    model = LogisticRegression(dataset=samples)
    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc", "f1"], output_path="output")
    trainer.train(dl(train, True), dl(val, False), epochs=a.epochs, monitor="roc_auc")
    model.print_weights()
    print(trainer.evaluate(dl(test, False)))
    


if __name__ == "__main__":
    main()
