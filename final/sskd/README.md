# Fine-grained long-tailed food image classification (ICCV'21 Workshop)
## Contribution [[Poster]](https://github.com/r09921135/dlcv/blob/master/final/sskd/poster.pdf)
* Bring the Self-Supervised Knowledge Distillation (SSKD) into the long-tailed problem to have a better representation learning. 
* Cumulative learning (CL) is adopted to avoid damaging the learned universal features when emphasizing the tail classes.
* Propose Image-wise Test-time Aggregation (ITA) to learn the aggregation weights of three experts image-wisely in test-time.

## Results
### Food_LT dataset
|    Methods   |  All acc. | Fre. acc. |   Comm. acc.  |   Rare acc.  |    
| :------------: | :--------: | :--------: | :--------: | :--------: |
|    [TADE](https://github.com/Vanint/TADE-AgnosticLT)   |   0.745   |    0.754    |   0.762   |   0.582  |
|     Ours     |   **0.792**   |    **0.821**    |   **0.792**   |   **0.587**  |

## Training
Run the following bash code, $1 below is the data directory:
    
    bash train_phase_reproduce.sh $1

After the training phase is complete, you can go to saved/models/Food_LT_ResNeXt50_SSKD/ to find our the best model.

## Evaluation
### Evaluation on the valid set
Run the following cmd, the performances on each track will be shown, $1 below is the data directory:
    
    python3 val_all_track.py -c test_time_config.json -r saved/models/Food_LT_ResNeXt50_SSKD/model_best.pth -w aggregation_weight.pth -p main_pred.csv -f $1
    
### Evaluation on the testing set
Run the following cmd, $1 below is the data directory:
    
    python3 test.py -c test_time_config.json -r saved/models/Food_LT_ResNeXt50_SSKD/model_best.pth -w aggregation_weight.pth -p main_pred.csv -f $1
    

