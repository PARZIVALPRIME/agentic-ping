# Queues the public-100 RESUME run (rows already in results/llm_full.json are
# skipped) as soon as the hidden-50 run (pid in results/_hidden_llm.pid) exits.
$hid = [int](Get-Content 'D:\gg\results\_hidden_llm.pid')
while (Get-Process -Id $hid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
Set-Location 'D:\gg'
cmd /c "python run_benchmark.py --out results/llm_full.json --summary results/llm_full_summary.json --resume > results\llm_full_resume.log 2>&1"
