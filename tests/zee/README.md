## Condor launch

Launch by executing the commands in the folder `tests/zee`:

```
chmod +x *.sh
```

then one-by-one

```
TASK_SCRIPT="gridtune_task_EEm.sh"; source submit.sh
TASK_SCRIPT="gridtune_task_EEp.sh"; source submit.sh
TASK_SCRIPT="gridtune_task_EB.sh";  source submit.sh
```

m.mieskolainen@imperial.ac.uk, 2024