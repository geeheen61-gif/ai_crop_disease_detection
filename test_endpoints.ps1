$baseUrl = "https://ai-crop-disease-detection.onrender.com"
$localImagePath = "$PSScriptRoot\temp_test_leaf.jpg"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Generate a synthetic plant image for testing (reliable)
try {
    Write-Host "Generating synthetic plant image..." -ForegroundColor Gray
    # RGB(150, 200, 50) -> Hue ~80 degrees. Inside [35, 85] range used by backend heuristic.
    python -c "from PIL import Image; img = Image.new('RGB', (200, 200), color=(150, 200, 50)); img.save(r'$localImagePath')"
    
    # NOTE: To test with a REAL image, comment out the python line above and set:
    # $localImagePath = "C:\Path\To\Your\Real\Leaf.jpg"
} catch {
    Write-Host "⚠️ Failed to generate image." -ForegroundColor Red
}


function Test-Endpoint {
    param($Title, $Uri, $Method = "Get", $Body = $null, $ContentType = $null)
    Write-Host "`n--- $Title ($Uri) ---" -ForegroundColor Cyan
    try {
        $params = @{ Uri = $Uri; Method = $Method }
        if ($Body) { $params.Body = $Body }
        if ($ContentType) { $params.ContentType = $ContentType }
        
        $response = Invoke-RestMethod @params
        return $response
    } catch {
        Write-Host "❌ Failed: $_" -ForegroundColor Red
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            if ($stream) {
                $reader = New-Object System.IO.StreamReader($stream)
                Write-Host "   Details: $($reader.ReadToEnd())" -ForegroundColor DarkRed
            }
        } catch {}
        return $null
    }
}

# 1. Health Check
$res = Test-Endpoint "Health Check" "$baseUrl/"
if ($res) { Write-Host "✅ Status: $($res.status)" -ForegroundColor Green }

# 2. Store Sensor Data
$timestamp = [int][double]::Parse((Get-Date -UFormat %s))
$sensorPayload = @{ temperature = 26.5; humidity = 62.0; soil = 48; rain = 4095; light = 150.0; ts = $timestamp } | ConvertTo-Json
$res = Test-Endpoint "Store Sensor Data" "$baseUrl/sensors/store" "Post" $sensorPayload "application/json"
if ($res) { Write-Host "✅ Stored: $($res.ok)" -ForegroundColor Green }

# 3. Get Latest Sensor Data
$res = Test-Endpoint "Get Latest Sensor Data" "$baseUrl/sensors/latest"
if ($res) { 
    Write-Host "✅ Received: Temp=$($res.temperature), Hum=$($res.humidity), TS=$($res.ts)" -ForegroundColor Green 
}

# 4. Sensor History
$res = Test-Endpoint "Sensor History" "$baseUrl/sensors/history?limit=5"
if ($res) {
    Write-Host "✅ History Count: $($res.readings.Count)" -ForegroundColor Green
    if ($res.readings.Count -gt 0) {
        $latest = $res.readings[0]
        Write-Host "   Latest History Item: Temp=$($latest.temperature), TS=$($latest.ts)" -ForegroundColor Gray
    }
}

# 5. Prediction (File Upload)
Write-Host "`n--- Prediction (File Upload) ($baseUrl/predict) ---" -ForegroundColor Cyan
if (Test-Path $localImagePath) {
    try {
        # PowerShell 5.1 doesn't have native multipart/form-data support in Invoke-RestMethod easily, 
        # so we use HttpClient for a reliable file upload test.
        Add-Type -AssemblyName System.Net.Http
        $client = New-Object System.Net.Http.HttpClient
        $content = New-Object System.Net.Http.MultipartFormDataContent
        
        $fileStream = [System.IO.File]::OpenRead($localImagePath)
        $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
        $content.Add($fileContent, "file", "test_leaf.jpg")
        
        $stringContent = New-Object System.Net.Http.StringContent("true")
        $content.Add($stringContent, "include_guidance")

        $responseTask = $client.PostAsync("$baseUrl/predict", $content)
        $responseTask.Wait()
        $response = $responseTask.Result
        
        if ($response.IsSuccessStatusCode) {
            $rawJson = $response.Content.ReadAsStringAsync().Result
            # Write-Host "   Raw Response: $rawJson" -ForegroundColor Gray
            $json = $rawJson | ConvertFrom-Json
            
            if ($json.is_plant -eq $false) {
                Write-Host "✅ Request Success, but 'Not a Plant':" -ForegroundColor Yellow
                Write-Host "   Message: $($json.message)" -ForegroundColor Yellow
                Write-Host "   Score: $($json.plant_score)" -ForegroundColor Gray
            } else {
                Write-Host "✅ Prediction Success!" -ForegroundColor Green
                Write-Host "   Category: $($json.category_top)" -ForegroundColor Yellow
                Write-Host "   Confidence: $($json.confidence)"
                if ($json.guidance) {
                    Write-Host "   Guidance: $($json.guidance.Substring(0, [math]::Min(60, $json.guidance.Length)))..."
                }
            }
        } else {
            Write-Host "❌ Failed Status: $($response.StatusCode)" -ForegroundColor Red
            Write-Host "   Reason: $($response.Content.ReadAsStringAsync().Result)" -ForegroundColor Red
        }
        $fileStream.Close()
        $client.Dispose()
    } catch {
        Write-Host "❌ Exception during upload: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ Skipping prediction test: Local image not found at $localImagePath" -ForegroundColor Yellow
}

# 6. Prediction History
$res = Test-Endpoint "Prediction History" "$baseUrl/predictions/history?limit=5"
if ($res) {
    Write-Host "✅ Prediction History Count: $($res.predictions.Count)" -ForegroundColor Green
    if ($res.predictions.Count -gt 0) {
        $p = $res.predictions[0]
        Write-Host "   Latest: $($p.disease) ($($p.confidence))" -ForegroundColor Gray
    }
}

Write-Host "`n--- All Tests Completed ---" -ForegroundColor Cyan
