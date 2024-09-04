# Universal-Model-Merge-Scripter
Creates CMM script that can directly executed on Kaggle from easy merge script

# How To Use
Create merge plan txt file, execute main.py and choose the file via Planned Text Path  
Insert Title of the script, download link of VAE, your CivitAI and HuggingFace API, User/Repo if you want to upload it on huggingface  
Press Save As Text, you'll get txt file  
Press Save As .ipynb, you'll get Jupiter Notebook file

# Merge Plan Script
`+Name, link of model`: download checkpoint model  
`+Name, link of model, %LR`: download LoRA model  
`-Name`: delete the model  
If the Name starts with **_** , this would automatically replaces to **TEMP_**  
`CM A + B alpha Result`: Weight Sum of 2 models saved as Result  
`CM A + B +S C alpha beta Result`: Sum Twice of 3 models saved as Result  
`CM A + B +T C alpha beta Result`: Triple Sum of 3 models saved as Result  
`CM A + B - C alpha Result`: Add Difference of 3 models saved as Result  
`LB Checkpoint A:alpha,B:beta,C:gamma Result`: Merge multiple LoRAs to Checkpoint, saved as Result  
'PR Checkpoint Result`: Prune the checkpoint and saved as Result

*Example* (Using the script for NovaFurry v3.0) :  
```+DHP, https://civitai.com/models/436585/dhxl-dead-horse-project-resources-sdxlpony
+MKF, https://civitai.com/models/135477/molkeun-furry-mix
+SXL, https://civitai.com/models/360745/sonicdiffusionxl

+ALP, https://civitai.com/models/471691/alphonse-white-datura-style-pony
+HYB, https://civitai.com/models/636191/hybridmixpony6
+MAG, https://civitai.com/models/562557?modelVersionId=626703

CM DHP + MKF +S SXL 0.2 0.3 _A

-DHP
-MKF
-SXL

CM ALP + HYB +S MAG 0.35 0.25 _B

-ALP
-HYB
-MAG

+FT, https://civitai.com/models/520960/furrytoonmix-pony
+IF, https://civitai.com/models/579632/indigo-furry-mix-xl
+DR, https://civitai.com/models/643746/drmfcgcyber

+OFA, https://civitai.com/models/349062?modelVersionId=494387
+LIL, https://civitai.com/models/582090/lilith-pony-toonmix
+WHP, https://civitai.com/models/575700/whipdxlmix

CM FT + IF +S DR 0.3 0.35 _C

-FT
-IF
-DR

CM OFA + LIL +S WHP 0.2 0.25 _D

-OFA
-LIL
-WHP

CM _A + _C +S _D 0.35 0.2 _E

-_A
-_C
-_D

CM _E + _B "0,1,1,1,1,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,1,1,1,1,1" NovaFurryV3
```  
____
