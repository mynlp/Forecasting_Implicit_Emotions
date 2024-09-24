# Forecasting Implicit Emotions Elicited in Conversations

This is the implementation code of:  
Yurie Koga, Shunsuke Kando, Yusuke Miyao. [Forecasting Implicit Emotions Elicited in Conversations.](https://aclanthology.org/2024.inlg-main.12/) INLG2024.

## Setup
 - Install Python 3.8.0
 - Run ```pip install -r requirements_distilroberta.txt``` for experiments with DistilRoBERTa,  
 and ```pip install -r requirements_llama2.txt``` for experiments with Llama 2.

## Dataset Preprocessing
### DailyDialog
You can download the original DailyDialog dataset from [here](http://yanran.li/dailydialog).  
Place the dataset in the `dailydialog` folder and run the following commands there.
```
python process_data.py
python process_data_next_emotion.py
```

### Reconstructed MEmoR


## Citation
```
@inproceedings{koga-etal-2024-forecasting-implicit,
    title = "Forecasting Implicit Emotions Elicited in Conversations",
    author = "Koga, Yurie  and
      Kando, Shunsuke  and
      Miyao, Yusuke",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.12",
    pages = "145--152",
}
```
