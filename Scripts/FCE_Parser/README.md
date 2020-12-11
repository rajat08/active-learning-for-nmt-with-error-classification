[![DOI](https://zenodo.org/badge/199615833.svg)](https://zenodo.org/badge/latestdoi/199615833)

## About
Cloned from https://github.com/gcunhase/FCECorpusXML



## FCE Corpus
* Cambridge Learner Corpus (CLC)'s First Certificate in English (FCE)
* 1,238 exam scripts taken in 2000 and 2001: 1,061 (train), 80 (dev), 97 (test)
* Dataset for error detection

## How to Make
### 1. Source and target
Download FCE Corpus and convert to source and target `.txt` files
```
python main_xml_to_txt.py --fce_xml_dir data/ --results_dir data/fce_txt/
```
  + `fce_xml_dir`: directory where the FCE `.xml` Corpus will be downloaded
  + `results_dir`: directory where the `.txt` data will be saved


## Acknowledgement
* Please star or fork if this code was useful for you. If you use it in a paper, please cite as:
  ```
  @software{cunha_sergio2019fce_xml2txt,
      author       = {Gwenaelle Cunha Sergio},
      title        = {{gcunhase/FCECorpusXML: Converting FCE Corpus from XML to TXT format}},
      month        = oct,
      year         = 2019,
      doi          = {10.5281/zenodo.3496455},
      version      = {1.0.0},
      publisher    = {Zenodo},
      url          = {https://github.com/gcunhase/FCECorpusXML}
      }
  ```

* *Disclaimer*: This is not my dataset, but I found the need to convert it to a different format for use with [this model](https://github.com/skasewa/wronging), thus the creation of this repository.

 

  * Please refer to the following citation if you use the FCE Corpus:  
    ```
    @inproceedings{yannakoudakis2011new,
      title={A new dataset and method for automatically grading ESOL texts},
      author={Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben},
      booktitle={Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1},
      pages={180--189},
      year={2011},
      organization={Association for Computational Linguistics}
    }
    ```
