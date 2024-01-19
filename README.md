## Semantic data processing and representation 
This repository holds all course project files for the THWS winter semester 2023/2024 course "Semantic data and Representation".
The course group members are:
- Bangguo Xu 5123723
- Simei Yan 5123720
- Liang Liu 5123719
- Zitai Wu 

## Final Project introduction
This project is a Transformer generative single-round dialog model built on a 50w Chinese xiaohuangji dialog corpus.The model was trained on a Tesla V100-32GB for 5 epochs in about 4 hours. This project is inspired by another [single-round dialog model](https://github.com/Schellings/Seq2SeqModel) built using the seq2seq model.

The results of the current model were not particularly satisfactory (the specific results are shown below), and need to be further improved.
<img src="https://github.com/xbgthws/Semantic-data-processing-and-representation/blob/main/4.%20Final%20Project/Chinese-Chatbot/chatbot.png" alt="" style="zoom:67%;" />

### Steps for using the model
#### 1.Generate the word list

```shell
python data_processing.py
```

#### 2.Training the model

```shell
python train.py
```

Tip: The parameter settings in `config.py` can be adjusted a bit before model training.The trained models are saved in the `saved_models` directory.

#### 3. Experience Interactive Chat (currently Chinese only)
```shell
python chat.py
```


