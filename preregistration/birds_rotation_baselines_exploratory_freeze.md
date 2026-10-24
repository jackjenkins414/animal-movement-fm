# Training-free baselines on the 30 rotation birds (exploratory)

Exploratory, never a tiered result. Written 24-10-2026, before any baseline was scored on the 27 birds below.

V, ARIMA, AVG3 and AVG10, computed exactly as the bird study computed them for the goose and dove, on the rotation's 6 h test windows, floor and scoring rules. Nothing is trained or tuned, and P is checked identical to the rotation's sealed P. The 27 birds outside fold 0 are new to these baselines, but their test partitions were already opened by the rotation's seal (30-09-2026), so no test partition is opened for the first time. Fold 0 (gull, goose, dove) is read from earlier sealed files, and its twelve values were known when this was written: all are below −0.05.

Scoring: travel p90 skill against P with the paired bootstrap (2,000 at seed 42), $\pm$ 0.05 as no difference. The travel mean, travel median and all window median are reported as point values, with an interval only on the all window mean. Evaluator: `experiments/reporting/bird_rotation_baselines_sealed_eval.py`. Its diagnostic on 24-10 found all 27 rotation P files and all 15 fold-0 files present, and read no test data.

Expectation: all four baselines are worse than persistence. Read on the 27 new birds, each baseline's median travel p90 skill against P is below −0.05.
