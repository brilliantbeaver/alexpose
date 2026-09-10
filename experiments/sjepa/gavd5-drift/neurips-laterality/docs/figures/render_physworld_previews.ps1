param([string[]]$Assets = @())
$ErrorActionPreference = 'Stop'
$figureDirectory = $PSScriptRoot
$chromeExecutable = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
if (-not (Test-Path -LiteralPath $chromeExecutable)) {
    $chromeExecutable = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
}
if (-not (Test-Path -LiteralPath $chromeExecutable)) {
    throw 'Install Chrome or Edge, or render the SVG files with another standards-compliant renderer.'
}
$figuresToRender = @(
    @{ Name = 'training_pipeline_compact' },
    @{ Name = 'training_pipeline' },
    @{ Name = 'reflection_and_target' },
    @{ Name = 'learning_results' }
)
foreach ($figure in $figuresToRender) {
    if ($Assets.Count -gt 0 -and $figure.Name -notin $Assets) { continue }
    $svgPath = Join-Path $figureDirectory ($figure.Name + '.svg')
    $pngPath = Join-Path $figureDirectory ($figure.Name + '.png')
    if (-not (Test-Path -LiteralPath $svgPath)) { throw "Missing vector source: $svgPath" }
    [xml]$svgDocument = Get-Content -LiteralPath $svgPath -Encoding UTF8 -Raw
    $svgViewBox = $svgDocument.DocumentElement.GetAttribute('viewBox') -split '\s+'
    if ($svgViewBox.Count -ne 4) { throw "Invalid SVG viewBox: $svgPath" }
    $previewWidth = [int][math]::Ceiling([double]$svgViewBox[2])
    $previewHeight = [int][math]::Ceiling([double]$svgViewBox[3])
    $previewProfile = Join-Path ([IO.Path]::GetTempPath()) ('physworld-svg-' + [guid]::NewGuid().ToString('N'))
    $previewArgs = @(
        '--headless=new', '--disable-gpu', '--no-first-run', '--hide-scrollbars',
        ('--user-data-dir="' + $previewProfile + '"'),
        ('--screenshot="' + $pngPath + '"'),
        ('--window-size=' + $previewWidth + ',' + $previewHeight),
        ([uri]$svgPath).AbsoluteUri
    )
    $previewProcess = Start-Process -FilePath $chromeExecutable -ArgumentList $previewArgs -WindowStyle Hidden -Wait -PassThru
    if ($previewProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pngPath)) {
        throw "Could not render $svgPath"
    }
    Write-Output "Rendered $($figure.Name).png"
}
