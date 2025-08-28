# run_analysis.ps1
# Run the flower-visit analyzer and produce a multi-panel PDF report + PNG plots.
# Edit $MetadataPath if you want to use a different metadata JSON.

$Python = ".\.venv\Scripts\python.exe"
$MetadataPath = ".\models\trajectory_autonomous_training_1_25000k_2025-08-27_083516.json"
$ReportPath = ".\report_v2.pdf"
$PlotsDir = ".\report_plots_v2"

# Ensure virtualenv python exists (adjust if needed)
if (-Not (Test-Path $Python)) {
    Write-Host "Python interpreter not found at $Python. Update the script to point to your python.exe." -ForegroundColor Yellow
    exit 1
}

# Create plots directory
if (-Not (Test-Path $PlotsDir)) { New-Item -ItemType Directory -Path $PlotsDir | Out-Null }

# Run analyzer: --augment will attempt to reconstruct trajectory if missing and will write it back to metadata
& $Python .\analyze_flower_visits.py $MetadataPath --save-report $ReportPath --save-plots $PlotsDir --augment

if ($LASTEXITCODE -eq 0) {
    Write-Host "Analysis completed. Report saved to $ReportPath and plots to $PlotsDir" -ForegroundColor Green
} else {
    Write-Host "Analyzer exited with code $LASTEXITCODE. Check output above for errors." -ForegroundColor Red
}
