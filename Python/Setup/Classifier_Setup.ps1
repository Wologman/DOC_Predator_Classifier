# This script should be run from the 'Setup' folder
$predatorEnvironment = 'cv_pytorch'
$predatorYamlFile = 'win_predator_env.yaml'
$environmentExists = (conda info --envs | Select-String -Pattern "^$predatorEnvironment").Count -gt 0


if ($environmentExists) {
    conda activate $predatorEnvironment
    conda list
    Write-Host "The Conda environment '$predatorEnvironment' already exists, you should not need to do anything further to run it. 
    To reinstall first run the following command in a terminal: conda env remove --name cv_pytorch"
    } else {   
    # Doesn't work on DOC LAN network.  Go home, or hotspot from a phone, or try the guest wifi network
    conda env create --file $predatorYamlFile
    conda activate $predatorEnvironment
    conda list
    Write-Host "The Conda environment '$megaEnvironment' was created"
    }

conda deactivate
Write-Host "The classifier environment setup (cv_pytorch) script has completed without error"