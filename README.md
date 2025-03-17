# GenerativeTopologyOptimization

1. Clone Diffusion-SDF: ```git clone https://github.com/princeton-computational-imaging/Diffusion-SDF.git```
2. Run the following:
```
pip install open3d
pip install einops_exts
pip install rotary_embedding_torch
cd utils
python objMaker.py
python csvMaker.py

```
3. Create the following file under the name specs_selto.json in config/stage1_sdf:
```
{
    "Description" : "training joint SDF-VAE model for modulating SDFs on the couch dataset",
    "DataSource" : "data",
    "GridSource" : "data/grid_data",
    "TrainSplit" : "data/splits/selto_all.json",
    "TestSplit" : "data/splits/selto_all.json",
    
    "training_task": "modulation",
  
    "SdfModelSpecs" : {
      "hidden_dim" : 512,
      "latent_dim" : 256,
      "pn_hidden_dim" : 128,
      "num_layers" : 9
    },

    "SampPerMesh" : 16000,
    "PCsize" : 1024,
  
    "num_epochs" : 100001,
    "log_freq" : 5000,
    "kld_weight" : 1e-5,
    "latent_std" : 0.25,
  
    "sdf_lr" : 1e-4
}
```
