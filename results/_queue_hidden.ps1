# Queues the hidden-50 run to start as soon as the public-100 run (pid in
# results/_llm_full.pid) exits. Logs to results/hidden_llm.log.
$pub = [int](Get-Content 'D:\gg\results\_llm_full.pid')
while (Get-Process -Id $pub -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
Set-Location 'D:\gg'
cmd /c "python run_benchmark.py questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl --out results/hidden_llm.json --summary results/hidden_llm_summary.json > results\hidden_llm.log 2>&1"
