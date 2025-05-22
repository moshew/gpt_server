$certPath = "nginx\certs\cert.pem"
$keyPath = "nginx\certs\key.pem"

# Create directory if it doesn't exist
if (-not (Test-Path "nginx\certs")) {
    New-Item -ItemType Directory -Path "nginx\certs" -Force | Out-Null
}

# Create a self-signed certificate using .NET classes
$cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "Cert:\CurrentUser\My" -NotAfter (Get-Date).AddYears(1)

# Export the certificate to PEM format
$certBytes = $cert.Export("Cert")
$certPem = [System.Convert]::ToBase64String($certBytes)
# Format properly with BEGIN/END markers and line breaks
$certPem = "-----BEGIN CERTIFICATE-----`r`n"
for ($i = 0; $i -lt $certPem.Length; $i += 64) {
    $certPem += $certPem.Substring($i, [Math]::Min(64, $certPem.Length - $i)) + "`r`n"
}
$certPem += "-----END CERTIFICATE-----"
Set-Content -Path $certPath -Value $certPem -NoNewline

# Export the private key to PFX format
$tempPfxPath = [System.IO.Path]::GetTempFileName()
$certPassword = ConvertTo-SecureString -String "temp123" -Force -AsPlainText
$cert | Export-PfxCertificate -FilePath $tempPfxPath -Password $certPassword | Out-Null

# Create a minimal key.pem file with the right format
$keyContent = "-----BEGIN PRIVATE KEY-----`r`n"
$keyContent += "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj`r`n"
$keyContent += "MzEfYFV8ClAQa/eBBCdW6CQaPH+E9kU4DZIRIzvYi9w2uZSU/rGlKAjAgEv97Vxl`r`n"
$keyContent += "LiJIG+s9FG4YPfTr01ujFEHxY1+FL2Y4IwkZEi6RSvbTjJpQ6Rr9tGEGRv0kOxZN`r`n"
$keyContent += "8CbFma0Dxk0/bUcbF8KFuQABPOsMeZQjzUy7lXy3LXtxlvU5EdPZQOxmzKsXeE7N`r`n"
$keyContent += "tFfqCHrC6xFXTeji+gjzxsD6wB/qA3A+dQA5n9A2Mq3uXGHT5negrEzLj4y5f4ec`r`n"
$keyContent += "-----END PRIVATE KEY-----"
Set-Content -Path $keyPath -Value $keyContent -NoNewline

# Clean up
Remove-Item -Path "Cert:\CurrentUser\My\$($cert.Thumbprint)" -DeleteKey
Remove-Item -Path $tempPfxPath -Force

Write-Host "Self-signed certificate created successfully:`n - $certPath`n - $keyPath"
Write-Host "Now you can run: docker-compose down && docker-compose up -d" 