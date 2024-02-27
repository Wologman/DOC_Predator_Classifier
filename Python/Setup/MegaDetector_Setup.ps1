
$newDirectoryName = "MegaDetector_Repo"
$modelsDirectoryName = "Models"
$modelFileName = "md_v5a.0.0.pt"
$modelURL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
$scriptDir = $PSScriptRoot
$parentPath = (Split-Path -Path $scriptDir -Parent)
$grandparentPath = (Split-Path -Path $parentPath -Parent)
$download = $true
$downloadModel = $true
$megaEnvironment = 'cameratraps-detector'
$environmentExists = (conda info --envs | Select-String -Pattern "^$megaEnvironment").Count -gt 0

# Combine the grandparent directory path and the new directory name
$newDirectoryPath = Join-Path -Path $grandparentPath -ChildPath $newDirectoryName
$modelsDirectoryPath = Join-Path -Path $grandparentPath -ChildPath $modelsDirectoryName
$modelFilePath =  Join-Path -Path $modelsDirectoryPath -ChildPath $modelFileName

# Check if the new directory already exists
if (Test-Path -Path $newDirectoryPath -PathType Container) {
    # Ask the user for confirmation to proceed
    $confirmation = Read-Host "The directory '$newDirectoryName' already exists. Do you want to overwrite it? (Y/N)"

    if ($confirmation -eq 'Y' -or $confirmation -eq 'y') {
        # Remove the existing directory
        Remove-Item -Path $newDirectoryPath -Recurse -Force
        # Create the new directory
        New-Item -Path $newDirectoryPath -ItemType Directory
        Write-Host "Directory created successfully."
    } else {
        Write-Host "Operation aborted."
        $download = $false
    }
} else {
    # Create the new directory if it doesn't exist
    New-Item -Path $newDirectoryPath -ItemType Directory
    Write-Host "Directory created successfully."
}

if ($download -eq $true) {
    Set-Location -Path $newDirectoryPath
    Write-Host "Changed directory to: $newDirectoryPath"
    git clone https://github.com/agentmorris/MegaDetector
    git clone https://github.com/ecologize/yolov5/
}

if ($environmentExists) {
    conda activate cameratraps-detector
    conda list
    Write-Host "The Conda environment '$megaEnvironment' already exists, you shouldn't need to do anything more to use it. 
    To reinstall, first run the following command in a terminal: conda env remove --name cameratraps-detector"
    } else {
    # Setup of the MegaDetector Python environment
    Set-Location -Path "$newDirectoryPath"
    Write-Host "Changed directory to: $newDirectoryPath"    
    # Doesn't work on DOC LAN network.  Go home, or hotspot from a phone, or try the guest wifi network
    conda env create --file MegaDetector\envs\environment-detector.yml
    conda activate cameratraps-detector
    conda list
    Write-Host "The Conda environment '$megaEnvironment' was created"
    }
Set-Location -Path $scriptDir
conda deactivate
Write-Host "The MegaDetector setup script has completed without error"



#Now download the models file

# Check if the models directory already exists
if (Test-Path -Path $modelsDirectoryPath -PathType Container) {
    # Ask the user for confirmation to proceed
    $confirmation = Read-Host "The directory '$modelsDirectoryName' already exists. Do you want to overwrite it? (Y/N)"

    if ($confirmation -eq 'Y' -or $confirmation -eq 'y') {
        # Remove the existing directory
        Remove-Item -Path $modelsDirectoryPath -Recurse -Force
        # Create the new directory
        New-Item -Path $modelsDirectoryPath -ItemType Directory
        Write-Host "Models Directory created successfully."
    } else {
        Write-Host "Operation aborted, you still need to create a Models directory then download a model file."
        $downloadModel = $false
    }
} else {
    # Create the new directory if it doesn't exist
    New-Item -Path $modelsDirectoryPath -ItemType Directory
    Write-Host "Models directory created successfully."
}

if ($downloadModel -eq $true) {
    Set-Location -Path $modelsDirectoryPath
    Write-Host "Changed directory to: $mpdelsDirectoryPath"
    Invoke-WebRequest -Uri $modelURL -OutFile $modelFilePath
}

